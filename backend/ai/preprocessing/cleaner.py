"""
ai/preprocessing/cleaner.py
----------------------------------------------------
Genkit AI - Text Cleaning Pipeline

Features
--------
✓ Lowercasing
✓ HTML tag removal
✓ URL removal
✓ Special character normalization
✓ Whitespace normalization
✓ Stopword removal (configurable)
✓ Rule-based lemmatization
✓ Contraction expansion
✓ Repeated character reduction

Author : Genkit AI
"""

import re
import logging
from typing import List, Set, Optional

logger = logging.getLogger("Genkit AI")

# ============================================================
# CONTRACTIONS
# ============================================================
_CONTRACTIONS: dict = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "mustn't": "must not",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "it's": "it is",
    "it'll": "it will",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "what's": "what is",
    "who's": "who is",
    "how's": "how is",
    "let's": "let us",
    "genkit's": "genkit",
}

# ============================================================
# STOPWORDS (for keyword extraction, NOT for query cleaning)
# ============================================================
_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "must", "can",
    "to", "of", "for", "in", "on", "at", "by", "with", "as", "into",
    "this", "that", "these", "those", "it", "its",
    "and", "or", "but", "if", "so", "yet", "nor",
    "just", "very", "really", "quite", "also",
}

# ============================================================
# LEMMATIZATION RULES (rule-based, no external library)
# ============================================================
# Suffix replacement rules: (pattern, replacement)
_LEMMA_RULES: List[tuple] = [
    # Verb endings
    (re.compile(r"(\w+)ing$"),      lambda m: m.group(1) if len(m.group(1)) > 3 else m.group(0)),
    (re.compile(r"(\w+)ied$"),      lambda m: m.group(1) + "y"),
    (re.compile(r"(\w+)ies$"),      lambda m: m.group(1) + "y"),
    (re.compile(r"(\w+)ed$"),       lambda m: m.group(1) if len(m.group(1)) > 3 else m.group(0)),
    # Noun plurals
    (re.compile(r"(\w+)ves$"),      lambda m: m.group(1) + "f"),
    (re.compile(r"(\w+)ies$"),      lambda m: m.group(1) + "y"),
    (re.compile(r"(\w+)s$"),        lambda m: m.group(1) if len(m.group(1)) > 3 else m.group(0)),
]

# Words that should NOT be lemmatized (exception list)
_LEMMA_EXCEPTIONS: Set[str] = {
    "news", "series", "species", "services", "genkit", "websites",
    "businesses", "processes", "this", "was", "is", "has", "does",
    "pricing", "marketing", "branding", "hosting", "development",
}

# ============================================================
# CLEANER
# ============================================================

class TextCleaner:
    """
    Production text cleaning pipeline for Genkit AI.

    Usage
    -----
    cleaned = cleaner.clean("What services do you offer???")
    # → "what services do you offer"
    """

    def __init__(self) -> None:
        # Pre-compiled regex patterns
        self._html_pattern = re.compile(r"<[^>]+>")
        self._url_pattern = re.compile(
            r"https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/\S*"
        )
        self._repeated_char_pattern = re.compile(r"(.)\1{3,}")
        self._repeated_punctuation = re.compile(r"([!?.]){2,}")
        self._whitespace_pattern = re.compile(r"\s+")
        self._non_alpha_pattern = re.compile(r"[^a-z0-9\s\'\-]")

    # ----------------------------------------------------------
    # Step 1: Expand contractions
    # ----------------------------------------------------------
    def _expand_contractions(self, text: str) -> str:
        text = text.lower()
        for contraction, expansion in _CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        return text

    # ----------------------------------------------------------
    # Step 2: Remove HTML
    # ----------------------------------------------------------
    def _remove_html(self, text: str) -> str:
        return self._html_pattern.sub(" ", text)

    # ----------------------------------------------------------
    # Step 3: Remove URLs
    # ----------------------------------------------------------
    def _remove_urls(self, text: str) -> str:
        return self._url_pattern.sub(" ", text)

    # ----------------------------------------------------------
    # Step 4: Normalize repeated characters
    # ----------------------------------------------------------
    def _normalize_repeated(self, text: str) -> str:
        # "soooo" → "soo", "!!!" → "!"
        text = self._repeated_char_pattern.sub(r"\1\1", text)
        text = self._repeated_punctuation.sub(r"\1", text)
        return text

    # ----------------------------------------------------------
    # Step 5: Remove special characters (keep letters, digits, spaces, apostrophes, hyphens)
    # ----------------------------------------------------------
    def _remove_special(self, text: str) -> str:
        return self._non_alpha_pattern.sub(" ", text)

    # ----------------------------------------------------------
    # Step 6: Collapse whitespace
    # ----------------------------------------------------------
    def _normalize_whitespace(self, text: str) -> str:
        return self._whitespace_pattern.sub(" ", text).strip()

    # ----------------------------------------------------------
    # Main clean (keeps stopwords — for query cleaning)
    # ----------------------------------------------------------
    def clean(self, text: str) -> str:
        """
        Standard cleaning: lowercase, remove HTML/URLs/special chars.
        Preserves stopwords (important for keeping query meaning intact).

        Parameters
        ----------
        text : str

        Returns
        -------
        str  Cleaned text.
        """
        if not text:
            return ""
        text = self._expand_contractions(text)
        text = self._remove_html(text)
        text = self._remove_urls(text)
        text = self._normalize_repeated(text)
        text = self._remove_special(text)
        text = self._normalize_whitespace(text)
        return text

    # ----------------------------------------------------------
    # Clean + remove stopwords (for keyword extraction / retrieval)
    # ----------------------------------------------------------
    def clean_for_retrieval(self, text: str) -> str:
        """
        Cleaning with stopword removal.
        Used when building retrieval queries.

        Parameters
        ----------
        text : str

        Returns
        -------
        str  Cleaned text without stopwords.
        """
        cleaned = self.clean(text)
        words = cleaned.split()
        filtered = [w for w in words if w not in _STOPWORDS and len(w) > 1]
        return " ".join(filtered)

    # ----------------------------------------------------------
    # Lemmatize a single word
    # ----------------------------------------------------------
    def lemmatize_word(self, word: str) -> str:
        """
        Apply rule-based lemmatization to a single word.

        Parameters
        ----------
        word : str  A single lowercase word.

        Returns
        -------
        str  Lemmatized form.
        """
        if word in _LEMMA_EXCEPTIONS:
            return word
        if len(word) <= 4:
            return word
        for pattern, replacement in _LEMMA_RULES:
            match = pattern.fullmatch(word)
            if match:
                result = replacement(match)
                if len(result) >= 3:
                    return result
        return word

    # ----------------------------------------------------------
    # Lemmatize a sentence
    # ----------------------------------------------------------
    def lemmatize(self, text: str) -> str:
        """
        Apply lemmatization to all words in a sentence.

        Parameters
        ----------
        text : str

        Returns
        -------
        str  Lemmatized text.
        """
        words = text.split()
        return " ".join(self.lemmatize_word(w) for w in words)

    # ----------------------------------------------------------
    # Full NLP pipeline (clean → remove stopwords → lemmatize)
    # ----------------------------------------------------------
    def full_pipeline(self, text: str) -> str:
        """
        Full NLP preprocessing pipeline:
        clean → remove stopwords → lemmatize.

        Used for building keyword-based retrieval queries.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        cleaned = self.clean_for_retrieval(text)
        return self.lemmatize(cleaned)

    # ----------------------------------------------------------
    # Extract keywords from text
    # ----------------------------------------------------------
    def extract_keywords(self, text: str, top_n: int = 8) -> List[str]:
        """
        Extract meaningful keywords from text.

        Parameters
        ----------
        text  : str
        top_n : int  Maximum keywords to return.

        Returns
        -------
        List[str]
        """
        processed = self.full_pipeline(text)
        words = processed.split()
        # Remove very short words
        keywords = [w for w in words if len(w) > 2]
        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique: List[str] = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique[:top_n]


# ============================================================
# Singleton
# ============================================================
cleaner = TextCleaner()

__all__ = ["TextCleaner", "cleaner"]
