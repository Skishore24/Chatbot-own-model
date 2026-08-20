"""
backend/tests/test_tokenizer.py
----------------------------------------------------
Unit tests for Byte-Fallback BPE Tokenizer.
"""

import unittest
import tempfile
from pathlib import Path

from app.llm.tokenizer import ByteFallbackBPETokenizer, SPECIAL_TOKENS


class TestByteFallbackBPETokenizer(unittest.TestCase):

    def setUp(self):
        self.corpus = [
            "Genkit provides web development, mobile apps, and custom AI models.",
            "Contact Genkit at contact@genkit.in for software development pricing.",
            "React, Next.js, Python, FastAPI, and PyTorch are core technologies.",
            "வணக்கம், Genkit AI உங்கள் நிறுவனத்திற்கான தனிப்பயன் மென்பொருள்.",
            "Special characters: 🚀 #AI @Genkit $100 100% test!",
        ]
        self.tokenizer = ByteFallbackBPETokenizer(vocab_size=200)
        self.tokenizer.train_on_corpus(self.corpus, target_vocab_size=200)

    def test_special_tokens_presence(self):
        """Verify all special tokens exist in vocabulary."""
        for tok in SPECIAL_TOKENS:
            self.assertIn(tok, self.tokenizer.encoder)
            idx = self.tokenizer.encoder[tok]
            self.assertEqual(self.tokenizer.decoder[idx], tok)

    def test_roundtrip_english(self):
        """Verify standard English sentences encode and decode accurately."""
        text = "Genkit builds modern web applications and mobile apps."
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded.strip(), text.strip())

    def test_roundtrip_unicode_and_tamil(self):
        """Verify unicode and non-Latin scripts encode and decode using byte fallback."""
        text = "வணக்கம் Genkit 🚀"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded.strip(), text.strip())

    def test_empty_and_whitespace(self):
        """Verify empty and whitespace strings."""
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_serialization(self):
        """Verify tokenizer save and load from disk."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            self.tokenizer.save(temp_path)
            loaded_tokenizer = ByteFallbackBPETokenizer()
            loaded_tokenizer.load(temp_path)

            test_str = "Testing tokenizer serialization roundtrip."
            self.assertEqual(
                self.tokenizer.encode(test_str),
                loaded_tokenizer.encode(test_str),
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
