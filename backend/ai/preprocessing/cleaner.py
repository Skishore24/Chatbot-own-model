"""
cleaner.py
----------------------------------------------------
Genkit AI - Advanced NLP Text Cleaner

Features
--------
- Lowercase conversion
- HTML removal
- URL removal
- Email removal
- Phone removal
- Emoji removal
- Special character removal
- Number normalization
- Whitespace normalization
- Repeated punctuation cleanup
- Repeated character normalization
- Safe text cleaning

Author : Genkit AI
"""

import re
import html
import string
from typing import List


class TextCleaner:

    def __init__(self):

        self.punctuation = string.punctuation

        self.url_pattern = re.compile(
            r"(https?://\S+|www\.\S+)",
            re.IGNORECASE,
        )

        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )

        self.phone_pattern = re.compile(
            r"(\+?\d[\d\s\-\(\)]{7,20})"
        )

        self.html_pattern = re.compile(
            r"<[^>]+>"
        )

        self.space_pattern = re.compile(
            r"\s+"
        )

        self.repeat_char_pattern = re.compile(
            r"(.)\1{2,}"
        )

        self.repeat_punctuation_pattern = re.compile(
            r"([!?.,])\1+"
        )

    # --------------------------------------------------

    def lowercase(self, text: str) -> str:

        if not text:
            return ""

        return text.lower()

    # --------------------------------------------------

    def remove_html(self, text: str) -> str:

        text = html.unescape(text)

        return self.html_pattern.sub(" ", text)

    # --------------------------------------------------

    def remove_urls(self, text: str) -> str:

        return self.url_pattern.sub(" ", text)

    # --------------------------------------------------

    def remove_email(self, text: str) -> str:

        return self.email_pattern.sub(" ", text)

    # --------------------------------------------------

    def remove_phone(self, text: str) -> str:

        return self.phone_pattern.sub(" ", text)

    # --------------------------------------------------

    def remove_emojis(self, text: str) -> str:

        return re.sub(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "]+",
            "",
            text,
        )

    # --------------------------------------------------

    def remove_special_characters(self, text: str) -> str:

        return re.sub(
            r"[^a-zA-Z0-9\s.,!?]",
            " ",
            text,
        )

    # --------------------------------------------------

    def normalize_numbers(self, text: str) -> str:

        return re.sub(
            r"\d+",
            lambda x: x.group(0),
            text,
        )

    # --------------------------------------------------

    def normalize_repeated_characters(self, text: str) -> str:

        return self.repeat_char_pattern.sub(
            r"\1\1",
            text,
        )

    # --------------------------------------------------

    def normalize_punctuation(self, text: str) -> str:

        return self.repeat_punctuation_pattern.sub(
            r"\1",
            text,
        )

    # --------------------------------------------------

    def remove_extra_spaces(self, text: str) -> str:

        return self.space_pattern.sub(
            " ",
            text,
        ).strip()

    # --------------------------------------------------

    def tokenize(self, text: str) -> List[str]:

        return text.split()

    # --------------------------------------------------

    def detokenize(self, tokens: List[str]) -> str:

        return " ".join(tokens)

    # --------------------------------------------------

    def clean(self, text: str) -> str:

        if text is None:
            return ""

        text = str(text)

        text = self.lowercase(text)

        text = self.remove_html(text)

        text = self.remove_urls(text)

        text = self.remove_email(text)

        text = self.remove_phone(text)

        text = self.remove_emojis(text)

        text = self.remove_special_characters(text)

        text = self.normalize_numbers(text)

        text = self.normalize_repeated_characters(text)

        text = self.normalize_punctuation(text)

        text = self.remove_extra_spaces(text)

        return text

    # --------------------------------------------------

    def normalize(self, text: str) -> str:

        return self.clean(text)

    # --------------------------------------------------

    def clean_tokens(self, text: str):

        cleaned = self.clean(text)

        return self.tokenize(cleaned)

    # --------------------------------------------------

    def sentence_length(self, text: str) -> int:

        return len(self.clean_tokens(text))

    # --------------------------------------------------

    def is_empty(self, text: str) -> bool:

        return len(self.clean(text)) == 0


# ==========================================================
# Global Cleaner
# ==========================================================

cleaner = TextCleaner()
