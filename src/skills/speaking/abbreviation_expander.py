"""
SOWTEE Abbreviation Expander
Context-aware expansion of letter abbreviations into full sentences.

Example: "i w t s" → "I want to sleep"
"""

import structlog
from dataclasses import dataclass
from typing import Any

from ...services.model_manager import get_model_manager
from ...services.memory import get_memory_service

logger = structlog.get_logger(__name__)


@dataclass
class ExpansionResult:
    """Result of abbreviation expansion."""
    abbreviation: str
    expansions: list[str]  # Ranked by confidence, most confident first
    confidences: list[float]
    selected_index: int = 0  # Currently selected expansion
    
    @property
    def primary(self) -> str:
        """Get the primary (most confident) expansion."""
        return self.expansions[0] if self.expansions else ""
    
    @property
    def alternatives(self) -> list[str]:
        """Get alternative expansions (excluding primary)."""
        return self.expansions[1:] if len(self.expansions) > 1 else []
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "abbreviation": self.abbreviation,
            "expansions": self.expansions,
            "confidences": self.confidences,
            "selected_index": self.selected_index,
            "primary": self.primary,
            "alternatives": self.alternatives,
        }


class AbbreviationExpander:
    """
    Context-aware abbreviation expansion service with learning.
    
    Uses AI to expand letter abbreviations like "i w t s" into 
    meaningful sentences based on:
    - Scene context (what the user sees)
    - Conversation history (what was just said)
    - User's past selections (learning from history)
    - Common communication patterns
    
    The expander learns from user selections to provide better
    suggestions over time, similar to the intent prediction service.
    """
    
    # Common abbreviation patterns for fallback
    COMMON_EXPANSIONS = {
        "i w": ["I want", "I will", "I wish"],
        "i n": ["I need", "I'm not"],
        "i a": ["I am", "I ate"],
        "i h": ["I have", "I'm hungry", "I'm happy"],
        "i w t": ["I want to", "I would like to"],
        "i w t s": ["I want to sleep", "I want to stop", "I want to stay"],
        "i w t e": ["I want to eat", "I want to exit"],
        "i w c": ["I want coffee", "I want cake"],
        "y": ["Yes"],
        "n": ["No"],
        "t": ["Thanks", "Thank you"],
        "p": ["Please"],
        "h": ["Help", "Hello"],
        "h m": ["Help me"],
        "c i h": ["Can I have"],
    }
    
    def __init__(self) -> None:
        self._model_manager = get_model_manager()
        self._memory_service = get_memory_service()
        # Cache for current session's visual context
        self._cached_scene: str | None = None
    
    def update_scene_context(self, scene_description: str) -> None:
        """Update the cached scene context from vision service."""
        self._cached_scene = scene_description
    
    async def expand(
        self,
        abbreviation: str,
        scene_description: str | None = None,
        conversation_context: str | None = None,
        custom_context: str | None = None,
        num_suggestions: int = 5,
        user_id: str | None = None,
    ) -> ExpansionResult:
        """
        Expand an abbreviation into possible sentences.
        
        Args:
            abbreviation: Space-separated letters like "i w t s"
            scene_description: Description of the current visual scene
            conversation_context: Recent conversation history
            custom_context: User-defined situation context (e.g., "Giving a speech at hackathon")
            num_suggestions: Number of expansions to generate
            user_id: User ID for personalized history lookup
            
        Returns:
            ExpansionResult with ranked expansions
        """
        abbrev_clean = abbreviation.lower().strip()
        
        # Use cached scene if not provided
        effective_scene = scene_description or self._cached_scene
        
        # Retrieve user's history for this abbreviation
        history_expansions: list[dict] = []
        if user_id:
            try:
                history_expansions = await self._memory_service.retrieve_abbreviation_history(
                    user_id=user_id,
                    abbreviation=abbrev_clean,
                    scene_description=effective_scene,
                    conversation_context=conversation_context,
                    limit=5,
                )
                if history_expansions:
                    logger.debug(
                        "Found abbreviation history",
                        abbreviation=abbrev_clean,
                        count=len(history_expansions),
                    )
            except Exception as e:
                logger.warning("Failed to retrieve abbreviation history", error=str(e))
        
        # Try AI expansion with history context
        try:
            expansions, confidences = await self._ai_expand(
                abbrev_clean,
                effective_scene,
                conversation_context,
                num_suggestions,
                history_expansions,
                custom_context,
            )
            
            if expansions:
                return ExpansionResult(
                    abbreviation=abbrev_clean,
                    expansions=expansions,
                    confidences=confidences,
                )
        except Exception as e:
            logger.warning("AI expansion failed, using fallback", error=str(e))
        
        # Fallback: prioritize history if available
        if history_expansions:
            history_based = [h["expansion"] for h in history_expansions[:num_suggestions]]
            if history_based:
                confidences = [0.9 - (i * 0.05) for i in range(len(history_based))]
                return ExpansionResult(
                    abbreviation=abbrev_clean,
                    expansions=history_based,
                    confidences=confidences,
                )
        
        # Final fallback to rule-based expansion
        expansions = self._fallback_expand(abbrev_clean, num_suggestions)
        confidences = [1.0 - (i * 0.1) for i in range(len(expansions))]
        
        return ExpansionResult(
            abbreviation=abbrev_clean,
            expansions=expansions,
            confidences=confidences,
        )
    
    async def store_selection(
        self,
        user_id: str,
        abbreviation: str,
        selected_expansion: str,
        scene_description: str | None = None,
        conversation_context: str | None = None,
    ) -> dict:
        """
        Store a user's expansion selection for learning.
        
        Call this when the user selects an expansion to enable
        the system to learn their preferences.
        
        Args:
            user_id: User identifier
            abbreviation: The abbreviation that was expanded
            selected_expansion: The expansion the user chose
            scene_description: Scene context when selection was made
            conversation_context: Conversation context
            
        Returns:
            Storage result with selection count
        """
        effective_scene = scene_description or self._cached_scene
        
        return await self._memory_service.store_abbreviation_selection(
            user_id=user_id,
            abbreviation=abbreviation.lower().strip(),
            selected_expansion=selected_expansion,
            scene_description=effective_scene,
            conversation_context=conversation_context,
        )
    
    async def _ai_expand(
        self,
        abbreviation: str,
        scene_description: str | None,
        conversation_context: str | None,
        num_suggestions: int,
        history_expansions: list[dict] | None = None,
        custom_context: str | None = None,
    ) -> tuple[list[str], list[float]]:
        """Use AI to expand the abbreviation with history context."""
        
        # Build rich context for the model
        context_parts = []
        
        # User-defined situation context - HIGHEST PRIORITY
        if custom_context:
            context_parts.append(f"""IMPORTANT - USER'S CURRENT SITUATION:
{custom_context}

This is the user's actual context right now. Generate sentences that are HIGHLY RELEVANT to this situation.
For example, if they're giving a speech at a hackathon, suggest things they might say in that context.""")
        
        # User's history - what they've said before with this abbreviation
        if history_expansions:
            history_lines = []
            for h in history_expansions[:5]:
                count = h.get("selection_count", 1)
                expansion = h.get("expansion", "")
                history_lines.append(f"- \"{expansion}\" (used {count}x)")
            
            if history_lines:
                context_parts.append(f"""USER'S PREVIOUS EXPANSIONS for this abbreviation:
{chr(10).join(history_lines)}

PRIORITIZE these if they fit the current context - the user has chosen them before!""")
        
        # Scene context - what the user can see
        if scene_description:
            context_parts.append(f"""CURRENT SCENE (what the user sees):
{scene_description}

The user may want to comment on, ask about, or respond to something they see.""")
        
        # Conversation context - critical for follow-ups
        if conversation_context:
            context_parts.append(f"""CONVERSATION HISTORY (most recent at bottom):
{conversation_context}

IMPORTANT: The user is likely responding to or following up on this conversation.
- If someone asked a question, the user probably wants to answer it
- If someone made a statement, the user may want to agree, disagree, or add to it
- Consider what would be natural to say next in this conversation""")
        
        if not context_parts:
            context_str = "No context available - suggest general common phrases."
        else:
            context_str = "\n\n".join(context_parts)
        
        # Get letters for validation
        letters = abbreviation.split()
        letters_str = ", ".join([f'"{l.upper()}"' for l in letters])
        
        prompt = f"""You are helping a person with limited mobility communicate. They type letter abbreviations where each letter is the first letter of a word they want to say.

ABBREVIATION TO EXPAND: "{abbreviation}"
REQUIRED LETTERS (in order): {letters_str}

CONTEXT:
{context_str}

YOUR TASK:
Generate {num_suggestions} sentences the user most likely wants to say, ranked by relevance to the conversation.

CRITICAL CONSTRAINTS:
- Each sentence MUST have EXACTLY {len(letters)} words
- Word 1 MUST start with "{letters[0].upper() if letters else '?'}"
- Word 2 MUST start with "{letters[1].upper() if len(letters) > 1 else '?'}"
- (Continue pattern for all {len(letters)} letters)

PRIORITIZE:
1. **Direct responses** to the last thing said in conversation (questions deserve answers!)
2. **Follow-up comments** that continue the conversation naturally
3. **Reactions** to what's happening in the scene
4. **Common phrases** that fit the abbreviation

Example: If someone asked "How are you feeling?" and abbreviation is "i a f":
- Good: "I am fine" (answers the question)
- Bad: "I ate food" (ignores the conversation)

Return ONLY a JSON array:
[{{"sentence": "...", "confidence": 0.95}}, ...]"""

        response = await self._model_manager.generate(
            task_type="abbreviation",
            prompt=prompt,
            system_prompt="You are an AAC communication assistant helping someone have natural conversations. Your job is to expand letter abbreviations into sentences that are RELEVANT to the ongoing conversation. ALWAYS prioritize responses that make sense in context - if someone asks a question, suggest answers. Each letter in the abbreviation represents the first letter of a word. Every sentence MUST have the exact word count matching the abbreviation, with each word starting with the corresponding letter. Respond with ONLY valid JSON.",
        )
        
        expansions, confidences = self._parse_ai_response(response)
        
        # Validate expansions match the abbreviation
        valid_expansions = []
        valid_confidences = []
        for exp, conf in zip(expansions, confidences):
            if self._validate_expansion(exp, abbreviation):
                valid_expansions.append(exp)
                valid_confidences.append(conf)
            else:
                logger.debug("Filtered invalid expansion", expansion=exp, abbreviation=abbreviation)
        
        return valid_expansions, valid_confidences
    
    def _validate_expansion(self, expansion: str, abbreviation: str) -> bool:
        """
        Validate that an expansion matches the abbreviation.
        
        Each word in the expansion must start with the corresponding letter
        from the abbreviation.
        
        Args:
            expansion: The expanded sentence (e.g., "I want to sleep")
            abbreviation: Space-separated letters (e.g., "i w t s")
            
        Returns:
            True if valid, False otherwise
        """
        letters = abbreviation.lower().split()
        words = expansion.split()
        
        # Must have same number of words as letters
        if len(words) != len(letters):
            return False
        
        # Each word must start with the corresponding letter
        for word, letter in zip(words, letters):
            if not word.lower().startswith(letter.lower()):
                return False
        
        return True
    
    def _parse_ai_response(self, response: str) -> tuple[list[str], list[float]]:
        """Parse the AI response into expansions and confidences."""
        import json
        import re
        
        try:
            text = response.strip()
            
            # Handle Qwen3's thinking mode - remove <think>...</think> blocks
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = text.strip()
            
            # Try to extract JSON from markdown code blocks
            if "```" in text:
                # Find JSON in code blocks
                json_match = re.search(r'```(?:json)?\s*([\[\{].*?[\]\}])\s*```', text, re.DOTALL)
                if json_match:
                    text = json_match.group(1)
            
            # Try to find JSON array anywhere in the text
            if not text.startswith('['):
                json_match = re.search(r'\[[\s\S]*?\]', text)
                if json_match:
                    text = json_match.group(0)
            
            text = text.strip()
            
            # Parse JSON
            data = json.loads(text)
            
            expansions = []
            confidences = []
            
            for item in data:
                if isinstance(item, dict):
                    sentence = item.get("sentence", item.get("text", item.get("expansion", "")))
                    confidence = float(item.get("confidence", item.get("score", 0.8)))
                    if sentence:
                        expansions.append(sentence)
                        confidences.append(confidence)
                elif isinstance(item, str):
                    # Simple string array
                    expansions.append(item)
                    confidences.append(0.8)
            
            if expansions:
                logger.debug("Parsed AI response", num_expansions=len(expansions))
                return expansions, confidences
            
            logger.warning("No expansions found in AI response")
            return [], []
            
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse AI response for abbreviation expansion", 
                          error=str(e), response_preview=response[:200])
            return [], []
        except Exception as e:
            logger.warning("Unexpected error parsing AI response", error=str(e))
            return [], []
    
    def _fallback_expand(self, abbreviation: str, num_suggestions: int) -> list[str]:
        """Fallback to rule-based expansion."""
        
        # Check exact match first
        if abbreviation in self.COMMON_EXPANSIONS:
            return self.COMMON_EXPANSIONS[abbreviation][:num_suggestions]
        
        # Try partial matches (prefixes)
        for pattern, expansions in self.COMMON_EXPANSIONS.items():
            if abbreviation.startswith(pattern):
                # Use the expansion and try to complete the rest
                base_expansions = expansions[:num_suggestions]
                return base_expansions
        
        # Generic fallback based on first letter
        first_letters = abbreviation.split()
        if first_letters:
            first = first_letters[0]
            generic = {
                "i": ["I want", "I need", "I am"],
                "c": ["Can I", "Could you"],
                "h": ["Help me", "Hello"],
                "y": ["Yes"],
                "n": ["No", "Need"],
                "t": ["Thank you", "The"],
                "p": ["Please"],
            }
            if first in generic:
                return generic[first][:num_suggestions]
        
        # Last resort
        return [f"({abbreviation})"]


# Singleton instance
_abbreviation_expander: AbbreviationExpander | None = None


def get_abbreviation_expander() -> AbbreviationExpander:
    """Get the abbreviation expander singleton."""
    global _abbreviation_expander
    if _abbreviation_expander is None:
        _abbreviation_expander = AbbreviationExpander()
    return _abbreviation_expander
