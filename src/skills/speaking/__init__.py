"""
SOWTEE Speaking Skill Package
The primary skill for letter-based text input and speech output.
"""

from .skill import SpeakingSkill
from .letter_cards import LetterCardSystem, CardState
from .abbreviation_expander import AbbreviationExpander, ExpansionResult

__all__ = [
    "SpeakingSkill",
    "LetterCardSystem",
    "CardState",
    "AbbreviationExpander",
    "ExpansionResult",
]
