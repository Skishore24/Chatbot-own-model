"""
backend/app/ai/tokenizer/tokenizer.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Byte-Fallback BPE Tokenizer Engine
Pure Python & PyTorch Native Tokenizer with 16,000 Vocab Size and 0% <unk> Emissions.
"""

import os
import json
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple, Union
import torch

from app.core.logger import logger
from app.core.config import settings


class TrieNode:
    """Trie Node for accelerated longest-prefix subword matching."""
    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.token_id: Optional[int] = None


class ByteFallbackBPETokenizer:
    """
    Enterprise Byte-Fallback BPE Tokenizer for Genkit AI v5.0.
    Guarantees 0% <unk> tokens by falling back to UTF-8 byte tokens (<0x00> to <0xFF>).
    """

    SPECIAL_TOKENS = [
        "<pad>",           # ID 0
        "<bos>",           # ID 1
        "<eos>",           # ID 2
        "<unk>",           # ID 3 (Reserved fallback, rarely used due to byte fallback)
        "<context_start>", # ID 4
        "<context_end>",   # ID 5
        "<query_start>",   # ID 6
        "<query_end>",     # ID 7
        "<thought_start>", # ID 8
        "<thought_end>",   # ID 9
        "<ans_start>",     # ID 10
        "<ans_end>",       # ID 11
    ]

    NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)
    NUM_BYTE_TOKENS = 256  # <0x00> through <0xFF>

    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self.bpe_merges: Dict[Tuple[str, str], int] = {}
        self.byte_encoder: Dict[int, int] = {}
        self.byte_decoder: Dict[int, int] = {}
        self.trie_root = TrieNode()
        self.is_trained = False

        self._initialize_base_vocab()

    def _initialize_base_vocab(self) -> None:
        """Initializes special control tokens and 256 UTF-8 byte tokens."""
        # 1. Add Special Control Tokens
        for idx, token in enumerate(self.SPECIAL_TOKENS):
            self.encoder[token] = idx
            self.decoder[idx] = token

        # 2. Add 256 Byte Tokens (<0x00> to <0xFF>)
        byte_start_id = self.NUM_SPECIAL_TOKENS
        for b in range(self.NUM_BYTE_TOKENS):
            token_id = byte_start_id + b
            byte_token = f"<0x{b:02X}>"
            self.encoder[byte_token] = token_id
            self.decoder[token_id] = byte_token
            self.byte_encoder[b] = token_id
            self.byte_decoder[token_id] = b

        # Register special tokens in Trie
        for token, token_id in self.encoder.items():
            self._insert_trie(token, token_id)

    def _insert_trie(self, token: str, token_id: int) -> None:
        """Inserts a subword string token into the prefix Trie for fast encoding."""
        node = self.trie_root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.token_id = token_id

    def normalize_text(self, text: str) -> str:
        """Applies Unicode NFKC normalization and whitespace standardization."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\r\n|\r", "\n", text)
        return text

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encodes a string into a list of token IDs using Byte-Fallback BPE.
        Guarantees 0% <unk> tokens by resolving unrecognized characters to byte tokens.
        """
        if not text:
            return [self.encoder["<bos>"], self.encoder["<eos>"]] if add_special_tokens else []

        normalized = self.normalize_text(text)
        token_ids: List[int] = []

        if add_special_tokens:
            token_ids.append(self.encoder["<bos>"])

        # Process text using Trie prefix matching & BPE merges
        idx = 0
        n = len(normalized)

        while idx < n:
            # Longest prefix match via Trie
            node = self.trie_root
            best_match_len = 0
            best_match_id = None
            curr_idx = idx

            while curr_idx < n and normalized[curr_idx] in node.children:
                node = node.children[normalized[curr_idx]]
                curr_idx += 1
                if node.token_id is not None:
                    best_match_len = curr_idx - idx
                    best_match_id = node.token_id

            if best_match_id is not None and best_match_len > 0:
                token_ids.append(best_match_id)
                idx += best_match_len
            else:
                # Byte-Fallback Strategy: Encode character as raw UTF-8 bytes
                char = normalized[idx]
                raw_bytes = char.encode("utf-8")
                for byte_val in raw_bytes:
                    byte_token_id = self.byte_encoder[byte_val]
                    token_ids.append(byte_token_id)
                idx += 1

        if add_special_tokens:
            token_ids.append(self.encoder["<eos>"])

        return token_ids

    def encode_tensor(
        self,
        text: str,
        max_length: int = 2048,
        device: Union[str, torch.device] = "cpu",
        add_special_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Encodes text directly into a padded 1D PyTorch Int64 Tensor.
        """
        token_ids = self.encode(text, add_special_tokens=add_special_tokens)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            pad_id = self.encoder["<pad>"]
            token_ids.extend([pad_id] * (max_length - len(token_ids)))

        return torch.tensor(token_ids, dtype=torch.long, device=device)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into a human-readable text string.
        Seamlessly converts byte tokens back to raw UTF-8 characters.
        """
        if not token_ids:
            return ""

        special_set = set(self.SPECIAL_TOKENS) if skip_special_tokens else set()
        decoded_parts: List[str] = []
        byte_buffer: bytearray = bytearray()

        for tid in token_ids:
            token_str = self.decoder.get(tid, "")

            # Check if special token
            if skip_special_tokens and token_str in special_set:
                continue

            # Check if byte token
            if tid in self.byte_decoder:
                byte_val = self.byte_decoder[tid]
                byte_buffer.append(byte_val)
            else:
                # Flush accumulated byte buffer if switching to string token
                if byte_buffer:
                    decoded_parts.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                decoded_parts.append(token_str)

        # Final flush
        if byte_buffer:
            decoded_parts.append(byte_buffer.decode("utf-8", errors="replace"))

        return "".join(decoded_parts)

    def train_on_corpus(self, sentences: List[str], target_vocab_size: int = 16000) -> None:
        """
        Trains the BPE vocabulary on a list of domain sentences until target_vocab_size is reached.
        """
        logger.info(f"Training Byte-Fallback BPE Tokenizer on {len(sentences)} sentences (Target Vocab: {target_vocab_size})...")

        # Step 1: Pre-tokenize into character lists
        word_counts: Dict[Tuple[str, ...], int] = {}
        for sentence in sentences:
            sentence = self.normalize_text(sentence)
            words = sentence.split()
            for word in words:
                chars = tuple(list(word))
                word_counts[chars] = word_counts.get(chars, 0) + 1

        # Step 2: Iterative BPE Merges
        num_merges = target_vocab_size - len(self.encoder)
        for i in range(max(0, num_merges)):
            pairs: Dict[Tuple[str, str], int] = {}
            for word, freq in word_counts.items():
                for j in range(len(word) - 1):
                    pair = (word[j], word[j + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_subword = best_pair[0] + best_pair[1]
            new_id = len(self.encoder)

            self.encoder[new_subword] = new_id
            self.decoder[new_id] = new_subword
            self.bpe_merges[best_pair] = new_id
            self._insert_trie(new_subword, new_id)

            # Apply merge to word counts dictionary
            new_word_counts: Dict[Tuple[str, ...], int] = {}
            for word, freq in word_counts.items():
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and word[j] == best_pair[0] and word[j + 1] == best_pair[1]:
                        new_word.append(new_subword)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_word_counts[tuple(new_word)] = freq
            word_counts = new_word_counts

        self.vocab_size = len(self.encoder)
        self.is_trained = True
        logger.info(f"BPE Tokenizer Training Complete! Final Vocab Size: {self.vocab_size}")

    def save(self, filepath: Optional[str] = None) -> None:
        """Saves trained tokenizer vocabulary and BPE merges to JSON."""
        save_path = filepath or str(settings.MODEL_DIR / "bpe_tokenizer_v5.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "encoder": self.encoder,
            "merges": [f"{p[0]}|||{p[1]}" for p in self.bpe_merges.keys()],
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved BPE Tokenizer to {save_path}")

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "ByteFallbackBPETokenizer":
        """Loads a trained Byte-Fallback BPE Tokenizer from JSON."""
        load_path = filepath or str(settings.MODEL_DIR / "bpe_tokenizer_v5.json")
        tok = cls()

        if not os.path.exists(load_path):
            logger.warning(f"Tokenizer checkpoint not found at {load_path}. Initializing default Byte-Fallback instance.")
            return tok

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok.vocab_size = data["vocab_size"]
        tok.encoder = data["encoder"]
        tok.decoder = {v: k for k, v in tok.encoder.items()}

        # Re-build Trie & BPE merges
        for subword, token_id in tok.encoder.items():
            tok._insert_trie(subword, token_id)

        for merge_str in data.get("merges", []):
            parts = merge_str.split("|||")
            if len(parts) == 2:
                pair = (parts[0], parts[1])
                if merge_str in tok.encoder:
                    tok.bpe_merges[pair] = tok.encoder[merge_str]

        tok.is_trained = True
        logger.info(f"Successfully loaded Byte-Fallback BPE Tokenizer from {load_path} (Vocab Size: {tok.vocab_size})")
        return tok


# Global Default Tokenizer Instance (auto-loads trained tokenizer if available)
default_tokenizer = (
    ByteFallbackBPETokenizer.load()
    if (settings.MODEL_DIR / "bpe_tokenizer_v5.json").exists()
    else ByteFallbackBPETokenizer()
)
