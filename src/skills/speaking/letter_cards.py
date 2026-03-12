"""
SOWTEE Letter Card System
5-card letter selection for efficient text input.

Card Layout:
- Card 1: A B C D E (5 letters)
- Card 2: F G H I J (5 letters)  
- Card 3: K L M N O (5 letters)
- Card 4: P Q R S T (5 letters)
- Card 5: U V W X Y Z (6 letters - X and Z grouped when spread)
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
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "selected_card_index": self.selected_card_index,
            "current_letters": self.current_letters,
            "typed_text": self.typed_text,
        }


class LetterCardSystem:
    """
    5-card letter selection system for AAC input.
    
    Flow:
    1. Display 5 cards with letter groups
    2. User selects a card → letters spread to 5 sections
    3. For 6-letter card (UVWXYZ), X and Z are grouped
    4. If X+Z section selected, create 2-letter final selection
    5. Letter selected → add to text field, reset to cards
    """
    
    # Card definitions: 5 cards with letter groups
    CARDS = [
        ["A", "B", "C", "D", "E"],      # Card 0
        ["F", "G", "H", "I", "J"],      # Card 1
        ["K", "L", "M", "N", "O"],      # Card 2
        ["P", "Q", "R", "S", "T"],      # Card 3
        ["U", "V", "W", "X", "Y", "Z"], # Card 4 (6 letters)
    ]
    
    # Display labels for cards (combined letters)
    CARD_LABELS = [
        "ABCDE",
        "FGHIJ",
        "KLMNO",
        "PQRST",
        "UVWXYZ",
    ]
    
    def __init__(self) -> None:
        self._state = CardState()
    
    @property
    def state(self) -> CardState:
        """Get current state."""
        return self._state
    
    def reset(self) -> CardState:
        """Reset to initial state (showing 5 cards)."""
        self._state = CardState()
        return self._state
    
    def get_cards(self) -> list[dict[str, Any]]:
        """
        Get the 5 main cards for display.
        
        Returns:
            List of card objects with index, letters, and label
        """
        return [
            {
                "index": i,
                "letters": self.CARDS[i],
                "label": self.CARD_LABELS[i],
                "letter_count": len(self.CARDS[i]),
            }
            for i in range(5)
        ]
    
    def select_card(self, card_index: int) -> CardState:
        """
        Select a card and spread its letters.
        
        Args:
            card_index: Index of the card (0-4)
            
        Returns:
            Updated state with letters spread
        """
        if card_index < 0 or card_index >= len(self.CARDS):
            raise ValueError(f"Invalid card index: {card_index}")
        
        letters = self.CARDS[card_index]
        
        # For 6-letter card, group X and Z together
        if len(letters) == 6:
            # Create 5 sections: U, V, W, X+Z, Y
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
        
        # Single letter selected - add to typed text and reset
        self._state.typed_text += selected.lower()
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
