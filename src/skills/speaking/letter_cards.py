"""
SOWTEE Letter Card System
Multi-language letter selection for efficient text input.

English Card Layout:
- Card 1: A B C D E (5 letters)
- Card 2: F G H I J (5 letters)  
- Card 3: K L M N O (5 letters)
- Card 4: P Q R S T (5 letters)
- Card 5: U V W X Y Z (6 letters - X and Z grouped when spread)

Arabic Card Layout (28 letters across 6 cards):
- Card 1: أ ب ت ث ج
- Card 2: ح خ د ذ ر
- Card 3: ز س ش ص ض
- Card 4: ط ظ ع غ ف
- Card 5: ق ك ل م ن
- Card 6: ه و ي
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CardLevel(str, Enum):
    """Current level in the card selection hierarchy."""
    CARDS = "cards"      # Showing 5 main cards
    LETTERS = "letters"  # Showing spread letters from selected card


@dataclass
class CardState:
    """State of the letter card selection system."""
    level: CardLevel = CardLevel.CARDS
    selected_card_index: int | None = None
    current_letters: list[str] = field(default_factory=list)
    typed_text: str = ""
    language: str = "en"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "selected_card_index": self.selected_card_index,
            "current_letters": self.current_letters,
            "typed_text": self.typed_text,
            "language": self.language,
        }


class LetterCardSystem:
    """
    Multi-language letter card selection system for AAC input.
    
    Flow:
    1. Display cards with letter groups (5 for English, 6 for Arabic)
    2. User selects a card → letters spread to individual sections
    3. For grouped letters (e.g. English XZ), show sub-options
    4. Letter selected → add to text field, reset to cards
    """
    
    # ── English card definitions ──
    CARDS_EN = [
        ["A", "B", "C", "D", "E"],      # Card 0
        ["F", "G", "H", "I", "J"],      # Card 1
        ["K", "L", "M", "N", "O"],      # Card 2
        ["P", "Q", "R", "S", "T"],      # Card 3
        ["U", "V", "W", "X", "Y", "Z"], # Card 4 (6 letters)
    ]
    CARD_LABELS_EN = ["ABCDE", "FGHIJ", "KLMNO", "PQRST", "UVWXYZ"]

    # ── Arabic card definitions (28 letters across 6 cards) ──
    CARDS_AR = [
        ["أ", "ب", "ت", "ث", "ج"],     # Card 0
        ["ح", "خ", "د", "ذ", "ر"],     # Card 1
        ["ز", "س", "ش", "ص", "ض"],     # Card 2
        ["ط", "ظ", "ع", "غ", "ف"],     # Card 3
        ["ق", "ك", "ل", "م", "ن"],     # Card 4
        ["ه", "و", "ي"],                # Card 5 (3 letters)
    ]
    CARD_LABELS_AR = [
        "أ ب ت ث ج",
        "ح خ د ذ ر",
        "ز س ش ص ض",
        "ط ظ ع غ ف",
        "ق ك ل م ن",
        "ه و ي",
    ]

    # Backward-compat aliases
    CARDS = CARDS_EN
    CARD_LABELS = CARD_LABELS_EN

    def __init__(self) -> None:
        self._state = CardState()
    
    @property
    def state(self) -> CardState:
        """Get current state."""
        return self._state

    def _cards_for_lang(self, language: str | None = None) -> tuple[list[list[str]], list[str]]:
        """Return (cards, labels) for the given language."""
        lang = language or self._state.language or "en"
        if lang.startswith("ar"):
            return self.CARDS_AR, self.CARD_LABELS_AR
        return self.CARDS_EN, self.CARD_LABELS_EN

    def reset(self, language: str | None = None) -> CardState:
        """Reset to initial state (showing cards)."""
        lang = language or self._state.language or "en"
        self._state = CardState(language=lang)
        return self._state
    
    def get_cards(self, language: str | None = None) -> list[dict[str, Any]]:
        """
        Get the main cards for display.
        
        Returns:
            List of card objects with index, letters, and label
        """
        cards, labels = self._cards_for_lang(language)
        return [
            {
                "index": i,
                "letters": cards[i],
                "label": labels[i],
                "letter_count": len(cards[i]),
            }
            for i in range(len(cards))
        ]
    
    def select_card(self, card_index: int, language: str | None = None) -> CardState:
        """
        Select a card and spread its letters.
        
        Args:
            card_index: Index of the card
            language: Language code
            
        Returns:
            Updated state with letters spread
        """
        cards, _ = self._cards_for_lang(language)
        if card_index < 0 or card_index >= len(cards):
            raise ValueError(f"Invalid card index: {card_index}")
        
        letters = cards[card_index]
        
        # For English 6-letter card, group X and Z together
        if len(letters) == 6 and (language or self._state.language or "en").startswith("en"):
            spread_letters = ["U", "V", "W", "XZ", "Y"]
        else:
            spread_letters = letters.copy()
        
        self._state.level = CardLevel.LETTERS
        self._state.selected_card_index = card_index
        self._state.current_letters = spread_letters
        
        return self._state
    
    def get_spread_letters(self) -> list[dict[str, Any]]:
        """
        Get the spread letters for the currently selected card.
        
        Returns:
            List of letter objects with index and display info
        """
        if self._state.level != CardLevel.LETTERS:
            return []
        
        return [
            {
                "index": i,
                "letter": letter,
                "display": letter,
                "is_grouped": len(letter) > 1,  # X+Z is grouped
            }
            for i, letter in enumerate(self._state.current_letters)
        ]
    
    def select_letter(self, letter_index: int) -> tuple[CardState, str | list[str]]:
        """
        Select a letter from the spread.
        
        Args:
            letter_index: Index of the letter in spread (0-4)
            
        Returns:
            Tuple of (updated state, selected letter(s))
            If grouped letters (XZ), returns list for further selection
        """
        if self._state.level != CardLevel.LETTERS:
            raise ValueError("Not in letter selection mode")
        
        if letter_index < 0 or letter_index >= len(self._state.current_letters):
            raise ValueError(f"Invalid letter index: {letter_index}")
        
        selected = self._state.current_letters[letter_index]
        
        # Check if this is a grouped selection (X+Z)
        if len(selected) > 1:
            # Need to show X and Z as separate options
            self._state.current_letters = list(selected)  # ["X", "Z"]
            return self._state, list(selected)
        
        # Single letter selected - add to typed text
        # Arabic has no case distinction, so don't lowercase
        lang = self._state.language or "en"
        self._state.typed_text += selected if lang.startswith("ar") else selected.lower()
        result_letter = selected
        
        # Reset to card view
        self._state.level = CardLevel.CARDS
        self._state.selected_card_index = None
        self._state.current_letters = []
        
        return self._state, result_letter
    
    def add_space(self) -> CardState:
        """Add a space to the typed text (used after abbreviations)."""
        if self._state.typed_text and not self._state.typed_text.endswith(" "):
            self._state.typed_text += " "
        return self._state
    
    def backspace(self) -> CardState:
        """Remove the last character from typed text."""
        if self._state.typed_text:
            self._state.typed_text = self._state.typed_text[:-1]
        return self._state
    
    def clear_text(self) -> CardState:
        """Clear all typed text."""
        self._state.typed_text = ""
        return self._state
    
    def get_typed_text(self) -> str:
        """Get the current typed text."""
        return self._state.typed_text
    
    def go_back(self) -> CardState:
        """Go back from letters to cards."""
        if self._state.level == CardLevel.LETTERS:
            self._state.level = CardLevel.CARDS
            self._state.selected_card_index = None
            self._state.current_letters = []
        return self._state
