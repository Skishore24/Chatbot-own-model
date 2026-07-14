"""
ai/tokenizer/tokenizer.py
----------------------------------------------------
Genkit AI - Production BPE-Lite Tokenizer

Own tokenizer for Genkit AI.
No HuggingFace
No SentencePiece
No External APIs

Author : Genkit AI
"""

import os
import re
import json
from collections import Counter
from typing import Dict, List, Optional


class BPETokenizer:
    """
    Lightweight Byte Pair Encoding tokenizer.

    Features
    --------
    • Own implementation
    • Train from text
    • Encode / Decode
    • Save / Load
    • Special Tokens
    """

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<s>"
    EOS = "</s>"
    WORD_END = "</w>"

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[tuple]] = None,
    ):

        self.vocab = vocab or {}
        self.merges = merges or []

        self.inverse_vocab = {
            v: k for k, v in self.vocab.items()
        }

        self.pad_token_id = self.vocab.get(self.PAD, 0)
        self.unk_token_id = self.vocab.get(self.UNK, 1)
        self.bos_token_id = self.vocab.get(self.BOS, 2)
        self.eos_token_id = self.vocab.get(self.EOS, 3)

    # =====================================================
    # TEXT TOKENIZER
    # =====================================================

    @staticmethod
    def split_words(text: str) -> List[str]:

        return re.findall(
            r"\w+|[^\w\s]",
            text.lower()
        )

    # =====================================================
    # TRAIN
    # =====================================================

    @classmethod
    def train(
        cls,
        texts: List[str],
        vocab_size: int = 3000,
        num_merges: int = 1000,
    ):
        """
        Train tokenizer from text corpus.
        """

        word_freq = Counter()

        for text in texts:

            for word in cls.split_words(text):

                chars = " ".join(list(word))
                chars += f" {cls.WORD_END}"

                word_freq[chars] += 1

        vocab = {

            cls.PAD: 0,
            cls.UNK: 1,
            cls.BOS: 2,
            cls.EOS: 3,
            cls.WORD_END: 4

        }

        for word in word_freq:

            for ch in word.split():

                if ch not in vocab:

                    vocab[ch] = len(vocab)

        merges = []

        splits = {
            word: word.split()
            for word in word_freq
        }

        # ----------------------------
        # Learn Merge Rules
        # ----------------------------

        for _ in range(num_merges):

            if len(vocab) >= vocab_size:
                break

            pair_freq = Counter()

            for word, freq in word_freq.items():

                tokens = splits[word]

                for i in range(len(tokens) - 1):

                    pair = (tokens[i], tokens[i + 1])

                    pair_freq[pair] += freq

            if not pair_freq:
                break

            best_pair = pair_freq.most_common(1)[0][0]

            merges.append(best_pair)

            new_token = "".join(best_pair)

            if new_token not in vocab:

                vocab[new_token] = len(vocab)

            left, right = best_pair

            for word in splits:

                tokens = splits[word]

                merged = []

                i = 0

                while i < len(tokens):

                    if (
                        i < len(tokens) - 1
                        and tokens[i] == left
                        and tokens[i + 1] == right
                    ):

                        merged.append(new_token)

                        i += 2

                    else:

                        merged.append(tokens[i])

                        i += 1

                splits[word] = merged

        return cls(vocab=vocab, merges=merges)

    # =====================================================
    # APPLY MERGES
    # =====================================================

    def _apply_merges(self, chars: List[str]) -> List[str]:

        tokens = chars[:]

        for left, right in self.merges:

            merged = []

            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):

                    merged.append(left + right)

                    i += 2

                else:

                    merged.append(tokens[i])

                    i += 1

            tokens = merged

        return tokens

    # =====================================================
    # ENCODE
    # =====================================================

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> List[int]:

        ids = []

        if add_special_tokens:
            ids.append(self.bos_token_id)

        words = self.split_words(text)

        for word in words:

            chars = list(word)

            chars.append(self.WORD_END)

            tokens = self._apply_merges(chars)

            for token in tokens:

                ids.append(
                    self.vocab.get(
                        token,
                        self.unk_token_id
                    )
                )

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    # =====================================================
    # DECODE
    # =====================================================

    def decode(
        self,
        ids: List[int]
    ) -> str:

        ignore = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id
        }

        tokens = []

        for idx in ids:

            idx = int(idx)

            if idx in ignore:
                continue

            token = self.inverse_vocab.get(
                idx,
                self.UNK
            )

            tokens.append(token)

        text = ""

        for token in tokens:

            text += token.replace(
                self.WORD_END,
                " "
            )

        text = re.sub(r"\s+", " ", text)

        return text.strip()

        # =====================================================
    # SAVE TOKENIZER
    # =====================================================

    def save_pretrained(
        self,
        save_dir: str
    ):

        os.makedirs(save_dir, exist_ok=True)

        data = {

            "vocab": self.vocab,

            "merges": self.merges

        }

        with open(
            os.path.join(save_dir, "bpe_tokenizer.json"),
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

    # =====================================================
    # LOAD TOKENIZER
    # =====================================================

    @classmethod
    def from_pretrained(
        cls,
        save_dir: str
    ):

        path = os.path.join(
            save_dir,
            "bpe_tokenizer.json"
        )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        merges = [

            tuple(item)

            for item in data["merges"]

        ]

        return cls(

            vocab=data["vocab"],

            merges=merges

        )

    # =====================================================
    # UTILITIES
    # =====================================================

    def token_to_id(self, token: str) -> int:

        return self.vocab.get(
            token,
            self.unk_token_id
        )

    def id_to_token(self, idx: int) -> str:

        return self.inverse_vocab.get(
            idx,
            self.UNK
        )

    def __contains__(self, token: str):

        return token in self.vocab

    def __len__(self):

        return len(self.vocab)

    def vocab_size(self):

        return len(self.vocab)
