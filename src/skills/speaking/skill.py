"""
SOWTEE Speaking Skill
Main skill class for the speaking/communication capability.
"""

import structlog
from typing import Any

from ..base_skill import BaseSkill, SkillInfo, SkillContext, SkillStatus
from .letter_cards import LetterCardSystem, CardState
from .abbreviation_expander import AbbreviationExpander, ExpansionResult, get_abbreviation_expander
from ...services.model_manager import get_model_manager

logger = structlog.get_logger(__name__)


class SpeakingSkill(BaseSkill):
    """
    Speaking skill for AAC communication.
    
    Features:
    - 5-card letter selection system
    - Context-aware abbreviation expansion
    - Sentence suggestions with alternatives
    - Integration with TTS for speech output
    """
    
    SKILL_ID = "speaking"
    
    def __init__(self) -> None:
        self._letter_system = LetterCardSystem()
        self._expander: AbbreviationExpander | None = None
        self._model_manager = get_model_manager()
        self._language: str = "en"
    
    def get_info(self) -> SkillInfo:
        """Return skill information."""
        return SkillInfo(
            skill_id=self.SKILL_ID,
            name="Speaking",
            description="Type using letter cards and speak with AI-powered sentence completion",
            icon="🗣️",
            status=SkillStatus.ACTIVE,
            version="1.0.0",
        )
    
    async def initialize(self) -> None:
        """Initialize the speaking skill."""
        self._expander = get_abbreviation_expander()
        logger.info("Speaking skill initialized")
    
    async def process(self, context: SkillContext, **kwargs) -> dict[str, Any]:
        """
        Process a speaking skill request.
        
        Supported actions:
        - get_cards: Get the 5 letter cards
        - select_card: Select a card and get spread letters
        - select_letter: Select a letter
        - expand: Expand the current abbreviation
        - get_suggestions: Get sentence suggestions
        - reset: Reset the letter system
        """
        action = kwargs.get("action", "get_cards")
        language = kwargs.get("language", "en")
        self._language = language
        
        if action == "get_cards":
            return self._handle_get_cards(language)
        elif action == "select_card":
            return self._handle_select_card(kwargs.get("card_index", 0), language)
        elif action == "select_letter":
            return self._handle_select_letter(kwargs.get("letter_index", 0))
        elif action == "expand":
            return await self._handle_expand(context)
        elif action == "get_suggestions":
            return await self._handle_get_suggestions(context, kwargs.get("count", 5))
        elif action == "reset":
            return self._handle_reset(language)
        elif action == "backspace":
            return self._handle_backspace()
        elif action == "add_space":
            return self._handle_add_space()
        elif action == "go_back":
            return self._handle_go_back(language)
        elif action == "get_state":
            return self._handle_get_state()
        else:
            return {"error": f"Unknown action: {action}"}
    
    def _handle_get_cards(self, language: str = "en") -> dict[str, Any]:
        """Get the letter cards."""
        return {
            "action": "get_cards",
            "cards": self._letter_system.get_cards(language),
            "state": self._letter_system.state.to_dict(),
        }
    
    def _handle_select_card(self, card_index: int, language: str = "en") -> dict[str, Any]:
        """Select a card and spread letters."""
        try:
            state = self._letter_system.select_card(card_index, language)
            return {
                "action": "select_card",
                "card_index": card_index,
                "spread_letters": self._letter_system.get_spread_letters(),
                "state": state.to_dict(),
            }
        except ValueError as e:
            return {"error": str(e)}
    
    def _handle_select_letter(self, letter_index: int) -> dict[str, Any]:
        """Select a letter from the spread."""
        try:
            state, selected = self._letter_system.select_letter(letter_index)
            
            # Check if we got a grouped selection (list) or single letter
            if isinstance(selected, list):
                return {
                    "action": "select_letter",
                    "letter_index": letter_index,
                    "grouped_options": selected,
                    "state": state.to_dict(),
                }
            else:
                return {
                    "action": "select_letter",
                    "letter_index": letter_index,
                    "selected_letter": selected,
                    "typed_text": state.typed_text,
                    "state": state.to_dict(),
                }
        except ValueError as e:
            return {"error": str(e)}
    
    async def _handle_expand(self, context: SkillContext) -> dict[str, Any]:
        """Expand the current abbreviation."""
        if not self._expander:
            self._expander = get_abbreviation_expander()
        
        typed_text = self._letter_system.get_typed_text()
        
        if not typed_text.strip():
            return {
                "action": "expand",
                "error": "No text to expand",
                "state": self._letter_system.state.to_dict(),
            }
        
        # Update expander's scene cache from context
        if context.scene_description:
            self._expander.update_scene_context(context.scene_description)
        
        # Format as space-separated letters for expansion
        abbreviation = " ".join(typed_text.strip())
        
        result = await self._expander.expand(
            abbreviation=abbreviation,
            scene_description=context.scene_description,
            conversation_context=context.get_recent_context_string(),
            user_id=context.user_id,  # Pass user_id for history lookup
        )
        
        return {
            "action": "expand",
            "abbreviation": abbreviation,
            "expansion": result.to_dict(),
            "state": self._letter_system.state.to_dict(),
        }
    
    async def _handle_get_suggestions(
        self, 
        context: SkillContext,
        count: int = 5,
    ) -> dict[str, Any]:
        """Get AI-powered sentence suggestions."""
        typed_text = self._letter_system.get_typed_text()
        
        prompt = f"""You are an AAC (Augmentative and Alternative Communication) assistant.

Based on the context, suggest {count} complete sentences the user might want to say.

TYPED LETTERS: {typed_text if typed_text else "(none)"}
SCENE: {context.scene_description or "Unknown"}
CONVERSATION:
{context.get_recent_context_string()}

Respond with a JSON array of sentence suggestions:
[
  {{"sentence": "suggested sentence", "confidence": 0.95}},
  ...
]

Make suggestions:
- Relevant to the conversation context
- Natural and conversational
- If responding to a question, include appropriate answers
- If typed letters exist, they should match word initials

Only respond with valid JSON."""

        try:
            response = await self._model_manager.generate(
                task_type="suggestions",
                prompt=prompt,
            )
            
            suggestions = self._parse_suggestions(response)
            
            return {
                "action": "get_suggestions",
                "suggestions": suggestions,
                "typed_text": typed_text,
                "state": self._letter_system.state.to_dict(),
            }
        except Exception as e:
            logger.error("Failed to get suggestions", error=str(e))
            return {
                "action": "get_suggestions",
                "suggestions": [],
                "error": str(e),
                "state": self._letter_system.state.to_dict(),
            }
    
    def _parse_suggestions(self, response: str) -> list[dict[str, Any]]:
        """Parse AI suggestions response."""
        import json
        
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse suggestions response")
            return []
    
    def _handle_reset(self, language: str = "en") -> dict[str, Any]:
        """Reset the letter system."""
        state = self._letter_system.reset(language)
        return {
            "action": "reset",
            "state": state.to_dict(),
        }
    
    def _handle_backspace(self) -> dict[str, Any]:
        """Remove last character."""
        state = self._letter_system.backspace()
        return {
            "action": "backspace",
            "typed_text": state.typed_text,
            "state": state.to_dict(),
        }
    
    def _handle_add_space(self) -> dict[str, Any]:
        """Add space to typed text."""
        state = self._letter_system.add_space()
        return {
            "action": "add_space",
            "typed_text": state.typed_text,
            "state": state.to_dict(),
        }
    
    def _handle_go_back(self, language: str = "en") -> dict[str, Any]:
        """Go back from letters to cards."""
        state = self._letter_system.go_back()
        return {
            "action": "go_back",
            "cards": self._letter_system.get_cards(language) if state.level.value == "cards" else None,
            "state": state.to_dict(),
        }
    
    def _handle_get_state(self) -> dict[str, Any]:
        """Get current state."""
        state = self._letter_system.state
        return {
            "action": "get_state",
            "cards": self._letter_system.get_cards(self._language) if state.level.value == "cards" else None,
            "spread_letters": self._letter_system.get_spread_letters() if state.level.value == "letters" else None,
            "state": state.to_dict(),
        }
