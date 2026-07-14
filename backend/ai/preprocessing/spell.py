"""
spell.py
----------------------------------------------------
Genkit AI - Advanced Spell Corrector

Features
--------
- Common spelling correction
- Multi-word correction
- Case-insensitive matching
- Whole-word replacement
- Suggest corrections
- Add/remove custom words
- Detect spelling mistakes
- No external libraries

Author : Genkit AI
"""

import re
from difflib import get_close_matches
from typing import Dict, List


class SpellCorrector:

    def __init__(self):

        self.dictionary: Dict[str, str] = {

            # Company
            "gen kit": "genkit",
            "genkt": "genkit",
            "gnekit": "genkit",
            "genkitt": "genkit",

            # Website
            "websit": "website",
            "websitee": "website",
            "web page": "website",
            "webpage": "website",

            # Chatbot
            "chat bot": "chatbot",
            "chatbt": "chatbot",
            "chatbo": "chatbot",

            # AI
            "artificial inteligence": "artificial intelligence",
            "artificial intellgence": "artificial intelligence",
            "artifical intelligence": "artificial intelligence",

            # ML
            "machin learning": "machine learning",
            "machine learnng": "machine learning",
            "machine learn": "machine learning",

            # Mobile
            "mobil": "mobile",
            "moblie": "mobile",
            "mobil app": "mobile app",

            # Automation
            "automtion": "automation",
            "automaton": "automation",

            # Services
            "servces": "services",
            "servise": "service",
            "servicee": "service",

            "portfoli": "portfolio",
            "contcat": "contact",
            "adress": "address",
            "emial": "email",
            "phne": "phone",

            "pricingg": "pricing",
            "prise": "price",
            "quatation": "quotation",

            "devloper": "developer",
            "sofware": "software",
            "aplication": "application",

            "logn": "login",
            "passwrod": "password",

            "hostng": "hosting",
            "reactjs": "react",
            "pyhton": "python",
            "fast api": "fastapi",
            "data base": "database",
            "mysqll": "mysql",
            "mangodb": "mongodb",
        }

        self._compile_patterns()

    # --------------------------------------------------

    def _compile_patterns(self):

        self.patterns = []

        for wrong, right in sorted(
            self.dictionary.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        ):

            pattern = re.compile(

                rf"\b{re.escape(wrong)}\b",

                flags=re.IGNORECASE,

            )

            self.patterns.append((pattern, right))

    # --------------------------------------------------

    def correct(self, text: str) -> str:

        if not text:
            return ""

        corrected = text

        for pattern, replacement in self.patterns:

            corrected = pattern.sub(

                replacement,

                corrected,

            )

        corrected = re.sub(

            r"\s+",

            " ",

            corrected,

        )

        return corrected.strip()

    # --------------------------------------------------

    def has_spelling_errors(
        self,
        text: str,
    ) -> bool:

        return self.correct(text) != text

    # --------------------------------------------------

    def suggest(
        self,
        word: str,
        limit: int = 3,
    ) -> List[str]:

        return get_close_matches(

            word.lower(),

            list(self.dictionary.keys()),

            n=limit,

            cutoff=0.65,

        )

    # --------------------------------------------------

    def add_word(
        self,
        wrong: str,
        correct: str,
    ):

        self.dictionary[wrong.lower()] = correct.lower()

        self._compile_patterns()

    # --------------------------------------------------

    def remove_word(
        self,
        wrong: str,
    ):

        wrong = wrong.lower()

        if wrong in self.dictionary:

            del self.dictionary[wrong]

            self._compile_patterns()

    # --------------------------------------------------

    def update_dictionary(
        self,
        words: Dict[str, str],
    ):

        self.dictionary.update(words)

        self._compile_patterns()

    # --------------------------------------------------

    def reset(self):

        self.__init__()

    # --------------------------------------------------

    def total_words(self) -> int:

        return len(self.dictionary)

    # --------------------------------------------------

    def all_words(self):

        return dict(self.dictionary)


# ==========================================================
# Global Singleton
# ==========================================================

spell_checker = SpellCorrector()
