"""
backend/tests/test_stream_decoder.py
----------------------------------------------------
Unit tests for StreamDecoder and byte-fallback UTF-8 streaming reconstruction.
"""

import unittest
from app.llm.tokenizer import ByteFallbackBPETokenizer, StreamDecoder


class TestStreamDecoder(unittest.TestCase):

    def setUp(self):
        self.tokenizer = ByteFallbackBPETokenizer(vocab_size=1000)

    def test_ascii_streaming(self):
        """Test streaming standard ASCII words."""
        decoder = self.tokenizer.create_stream_decoder(skip_special_tokens=True)
        text = "Hello world from Genkit!"
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        streamed_parts = []
        for tid in token_ids:
            chunk = decoder.put(tid)
            if chunk:
                streamed_parts.append(chunk)

        remaining = decoder.flush()
        if remaining:
            streamed_parts.append(remaining)

        reconstructed = "".join(streamed_parts)
        self.assertEqual(reconstructed, text)

    def test_multibyte_unicode_and_tamil_streaming(self):
        """Test streaming multi-byte UTF-8 Tamil and emoji characters through byte fallback."""
        decoder = self.tokenizer.create_stream_decoder(skip_special_tokens=True)
        text = "Genkit AI தமிழ் 🚀 வணக்கம்"
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        streamed_parts = []
        for tid in token_ids:
            chunk = decoder.put(tid)
            if chunk:
                # Must never emit raw replacement characters during valid multi-byte decoding
                self.assertNotIn("\ufffd", chunk)
                streamed_parts.append(chunk)

        remaining = decoder.flush()
        if remaining:
            streamed_parts.append(remaining)

        reconstructed = "".join(streamed_parts)
        self.assertEqual(reconstructed, text)

    def test_special_tokens_handling(self):
        """Test that special tokens are properly skipped or kept based on configuration."""
        decoder_skip = self.tokenizer.create_stream_decoder(skip_special_tokens=True)
        decoder_keep = self.tokenizer.create_stream_decoder(skip_special_tokens=False)

        text = "Hello"
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)

        res_skip = "".join([decoder_skip.put(t) for t in token_ids] + [decoder_skip.flush()])
        res_keep = "".join([decoder_keep.put(t) for t in token_ids] + [decoder_keep.flush()])

        self.assertEqual(res_skip, "Hello")
        self.assertIn("<bos>", res_keep)
        self.assertIn("<eos>", res_keep)


if __name__ == "__main__":
    unittest.main()
