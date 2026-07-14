"""
keywords.py
----------------------------------------------------
Genkit AI - Advanced Keyword Extractor

Features
--------
- Text cleaning
- Tokenization
- Stop-word removal
- Frequency-based keyword extraction
- Company keyword detection
- Keyword scoring
- Bigram extraction
- Intent keyword detection
- Duplicate removal

Author : Genkit AI
"""

import re
from collections import Counter
from typing import List, Dict


class KeywordExtractor:

    def __init__(self):

        self.stop_words = {

            "a","an","the","is","am","are","was","were",
            "be","been","being","to","of","for","and",
            "or","but","if","then","this","that","these",
            "those","i","me","my","mine","you","your",
            "yours","he","she","it","they","them","their",
            "we","our","ours","in","on","at","by","with",
            "about","can","could","would","should","will",
            "shall","may","might","do","does","did","have",
            "has","had","what","which","who","when","where",
            "why","how","please","want","need","tell","give",
            "show","get","know","more","info","information"
        }

        self.company_keywords = {

            "genkit",
            "website",
            "web",
            "chatbot",
            "ai",
            "artificial",
            "intelligence",
            "automation",
            "machine",
            "learning",
            "python",
            "django",
            "fastapi",
            "react",
            "flutter",
            "mobile",
            "android",
            "ios",
            "software",
            "application",
            "portfolio",
            "service",
            "services",
            "pricing",
            "price",
            "quotation",
            "quote",
            "contact",
            "email",
            "phone",
            "marketing",
            "seo",
            "hosting",
            "cloud",
            "erp",
            "crm",
            "dashboard",
            "api"
        }

    # ------------------------------------------------

    def clean(self, text: str) -> str:

        if not text:
            return ""

        text = text.lower()

        text = re.sub(r"[^a-z0-9\s]", " ", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ------------------------------------------------

    def tokenize(self, text: str) -> List[str]:

        return self.clean(text).split()

    # ------------------------------------------------

    def remove_stop_words(
        self,
        words: List[str],
    ) -> List[str]:

        result = []

        for word in words:

            if len(word) < 2:
                continue

            if word in self.stop_words:
                continue

            result.append(word)

        return result

    # ------------------------------------------------

    def extract(
        self,
        text: str,
        top_k: int = 10,
    ) -> List[str]:

        words = self.tokenize(text)

        words = self.remove_stop_words(words)

        counts = Counter(words)

        return [

            word

            for word, _ in counts.most_common(top_k)

        ]

    # ------------------------------------------------

    def extract_with_scores(
        self,
        text: str,
        top_k: int = 10,
    ) -> Dict[str, int]:

        words = self.remove_stop_words(

            self.tokenize(text)

        )

        return dict(

            Counter(words).most_common(top_k)

        )

    # ------------------------------------------------

    def company_terms(
        self,
        text: str,
    ) -> List[str]:

        return [

            word

            for word in self.extract(text)

            if word in self.company_keywords

        ]

    # ------------------------------------------------

    def keyword_score(
        self,
        text: str,
    ) -> int:

        score = 0

        for word in self.extract(text):

            if word in self.company_keywords:

                score += 1

        return score

    # ------------------------------------------------

    def extract_bigrams(
        self,
        text: str,
        top_k: int = 5,
    ) -> List[str]:

        words = self.remove_stop_words(

            self.tokenize(text)

        )

        bigrams = []

        for i in range(len(words) - 1):

            bigrams.append(

                words[i] + " " + words[i + 1]

            )

        counts = Counter(bigrams)

        return [

            bigram

            for bigram, _ in counts.most_common(top_k)

        ]

    # ------------------------------------------------

    def detect_intent_keywords(
        self,
        text: str,
    ) -> List[str]:

        intents = {

            "pricing": [
                "price",
                "pricing",
                "cost",
                "quotation",
                "quote",
                "budget",
            ],

            "support": [
                "support",
                "error",
                "issue",
                "bug",
                "problem",
                "help",
            ],

            "contact": [
                "contact",
                "phone",
                "email",
                "address",
                "office",
            ],

            "services": [
                "website",
                "chatbot",
                "software",
                "application",
                "mobile",
                "ai",
                "automation",
            ],

        }

        words = set(

            self.extract(text)

        )

        result = []

        for intent, kws in intents.items():

            if words.intersection(kws):

                result.append(intent)

        return result

    # ------------------------------------------------

    def unique_keywords(
        self,
        text: str,
    ) -> List[str]:

        return list(

            dict.fromkeys(

                self.extract(text)

            )

        )

    # ------------------------------------------------

    def is_company_query(
        self,
        text: str,
    ) -> bool:

        return self.keyword_score(text) > 0


# ==========================================================
# Global Singleton
# ==========================================================

keyword_extractor = KeywordExtractor()
