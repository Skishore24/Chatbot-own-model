"""
backend/app/llm/tokenizer.py
----------------------------------------------------
Deterministic Byte-Fallback BPE Tokenizer for Genkit AI V6.
Pure Python implementation:
- 10 special tokens
- 256 byte tokens (<0x00> .. <0xFF>) for 100% UTF-8 safety
- Learned BPE merges
- Encode / Decode round-trip
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<context_start>",
    "<context_end>",
    "<query_start>",
    "<query_end>",
    "<ans_start>",
    "<ans_end>",
]

# Standard Unicode pre-tokenization regex for Python built-in re
PRE_TOKENIZE_REGEX = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _get_stats(words: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    """Computes adjacent pair frequencies across pre-tokenized words."""
    pairs: Dict[Tuple[str, str], int] = {}
    for word, freq in words.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


def _merge_word(word: Tuple[str, ...], pair: Tuple[str, str], merged_str: str) -> Tuple[str, ...]:
    """Replaces occurrences of pair in a single word tuple with merged_str."""
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(merged_str)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


class ByteFallbackBPETokenizer:
    """Byte-Fallback Byte-Pair Encoding (BPE) Tokenizer."""

    def __init__(self, vocab_size: int = 10000):
        self.target_vocab_size = vocab_size
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._special_set: Set[str] = set(SPECIAL_TOKENS)

        self._init_base_vocabulary()

    def _init_base_vocabulary(self) -> None:
        """Initializes special tokens and 256 byte-fallback tokens."""
        self.encoder = {}
        self.decoder = {}

        # 1. Special Tokens
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.encoder[token] = idx
            self.decoder[idx] = token

        # 2. Byte Tokens (<0x00> .. <0xFF>)
        for b in range(256):
            token = f"<0x{b:02X}>"
            idx = len(self.encoder)
            self.encoder[token] = idx
            self.decoder[idx] = token

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    @property
    def pad_id(self) -> int:
        return self.encoder.get("<pad>", 0)

    @property
    def bos_id(self) -> int:
        return self.encoder.get("<bos>", 1)

    @property
    def eos_id(self) -> int:
        return self.encoder.get("<eos>", 2)

    @property
    def unk_id(self) -> int:
        return self.encoder.get("<unk>", 3)

    def train_on_corpus(self, sentences: List[str], target_vocab_size: Optional[int] = None) -> None:
        """Trains BPE merges on text corpus up to target vocabulary size."""
        target_size = target_vocab_size or self.target_vocab_size
        self._init_base_vocabulary()
        self.merges = []

        # Count word frequencies using basic whitespace/punctuation pre-tokenization
        word_freqs: Dict[Tuple[str, ...], int] = {}
        for sentence in sentences:
            words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
            for word in words:
                # Convert characters to characters
                char_tuple = tuple(list(word))
                if char_tuple:
                    word_freqs[char_tuple] = word_freqs.get(char_tuple, 0) + 1

        # Also add space prefix tokens for words
        for sentence in sentences:
            tokens = sentence.split()
            for token in tokens:
                char_tuple = tuple([" "] + list(token)) if token else tuple()
                if char_tuple:
                    word_freqs[char_tuple] = word_freqs.get(char_tuple, 0) + 1

        # Add single character tokens to vocabulary first
        all_chars = set()
        for word in word_freqs.keys():
            for ch in word:
                all_chars.add(ch)

        for ch in sorted(all_chars):
            if ch not in self.encoder:
                idx = len(self.encoder)
                self.encoder[ch] = idx
                self.decoder[idx] = ch

        # Iteratively find most frequent pairs and merge
        num_merges = max(0, target_size - len(self.encoder))
        for _ in range(num_merges):
            stats = _get_stats(word_freqs)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2:
                # Pair occurred only once, stop merging
                break

            merged_token = best_pair[0] + best_pair[1]
            idx = len(self.encoder)
            self.encoder[merged_token] = idx
            self.decoder[idx] = merged_token
            self.merges.append(best_pair)

            # Update word frequencies
            new_word_freqs: Dict[Tuple[str, ...], int] = {}
            for word, freq in word_freqs.items():
                new_word = _merge_word(word, best_pair, merged_token)
                new_word_freqs[new_word] = new_word_freqs.get(new_word, 0) + freq
            word_freqs = new_word_freqs

    def _bpe_encode_word(self, word: str) -> List[str]:
        """Applies learned BPE merges to a single word."""
        if not word:
            return []
        tokens = list(word)
        for pair in self.merges:
            merged_str = pair[0] + pair[1]
            tokens = list(_merge_word(tuple(tokens), pair, merged_str))
            if len(tokens) <= 1:
                break
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes text into token ID sequence with byte fallback."""
        if not text:
            return []

        token_ids: List[int] = []
        if add_special_tokens:
            token_ids.append(self.bos_id)

        # Process text using word splitting while preserving special tokens
        parts = re.split(r"(<[a-zA-Z0-9_]+>)", text)
        for part in parts:
            if not part:
                continue
            if part in self._special_set and part in self.encoder:
                token_ids.append(self.encoder[part])
                continue

            # Tokenize regular text
            words = re.findall(r"\w+|[^\w\s]|\s+", part, re.UNICODE)
            for word in words:
                subwords = self._bpe_encode_word(word)
                for sw in subwords:
                    if sw in self.encoder:
                        token_ids.append(self.encoder[sw])
                    else:
                        # Byte fallback for unseen characters/symbols
                        for b in sw.encode("utf-8"):
                            byte_tok = f"<0x{b:02X}>"
                            token_ids.append(self.encoder.get(byte_tok, self.unk_id))

        if add_special_tokens:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs into string with byte-fallback reconstruction."""
        if not token_ids:
            return ""

        byte_stream: bytearray = bytearray()
        result_parts: List[str] = []

        def flush_bytes():
            nonlocal byte_stream, result_parts
            if byte_stream:
                result_parts.append(byte_stream.decode("utf-8", errors="replace"))
                byte_stream = bytearray()

        for tid in token_ids:
            tok = self.decoder.get(tid)
            if tok is None:
                continue

            if tok in self._special_set:
                flush_bytes()
                if not skip_special_tokens:
                    result_parts.append(tok)
                continue

            # Check if byte token (<0xHH>)
            if tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6:
                try:
                    b = int(tok[3:5], 16)
                    byte_stream.append(b)
                    continue
                except ValueError:
                    pass

            # Regular text token
            flush_bytes()
            result_parts.append(tok)

        flush_bytes()
        return "".join(result_parts)

    def save(self, filepath: str) -> None:
        """Saves tokenizer state to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab_size": len(self.encoder),
            "target_vocab_size": self.target_vocab_size,
            "encoder": self.encoder,
            "vocab": self.encoder,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str) -> None:
        """Loads tokenizer state from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {filepath}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.encoder = data.get("encoder") or data.get("vocab", {})
        self.decoder = {int(v): k for k, v in self.encoder.items()}
        self.target_vocab_size = data.get("vocab_size", data.get("target_vocab_size", len(self.encoder)))
        self.merges = [tuple(m) for m in data.get("merges", [])]
        self._special_set = set(SPECIAL_TOKENS)


# Default Singleton Tokenizer
default_tokenizer = ByteFallbackBPETokenizer()
