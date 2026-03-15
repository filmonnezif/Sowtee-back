"""
SOWTEE Predictive Text Suggestion Service
Provides continuous, context-aware word and sentence predictions.

Instead of abbreviation expansion, this service predicts what the user
wants to say based on:
- Visual scene (camera)
- Conversation history (what others said, what user said)
- User's typed text so far
- User vocabulary/history from memory
"""

import json
import time
import structlog
from uuid import uuid4

from ..config import get_settings
from .model_manager import get_model_manager, log_model_input, log_model_output, log_model_error
from .memory import get_memory_service
from .vision import get_vision_service

logger = structlog.get_logger(__name__)


class PredictedText:
    """A single text prediction."""
    def __init__(
        self,
        text: str,
        confidence: float = 0.7,
        is_completion: bool = False,
    ):
        self.text = text
        self.confidence = confidence
        self.is_completion = is_completion  # True = completes partial text

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "is_completion": self.is_completion,
        }


class PredictionResult:
    """Result of a prediction request."""
    def __init__(
        self,
        suggestions: list[PredictedText],
        ghost_text: str | None = None,
        processing_time_ms: float = 0.0,
    ):
        self.suggestions = suggestions
        self.ghost_text = ghost_text
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> dict:
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "ghost_text": self.ghost_text,
            "processing_time_ms": self.processing_time_ms,
        }


class PredictiveSuggestionService:
    """
    Context-aware predictive text service for AAC communication.
    
    Generates continuous predictions combining:
    - Visual scene context
    - Conversation history
    - User's partial text input
    - Learned vocabulary/preferences from memory
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._memory = get_memory_service()
        self._vision = get_vision_service()
        self._model_manager = get_model_manager()

        # Cache
        self._cached_scene: str | None = None
        logger.info("Predictive suggestions configured (using ModelManager with fallback)")

    async def get_predictions(
        self,
        user_id: str,
        partial_text: str = "",
        scene_description: str | None = None,
        conversation_history: list[dict] | None = None,
        num_suggestions: int = 5,
        language: str = "en",
    ) -> PredictionResult:
        """
        Get predictive text suggestions.

        Args:
            user_id: User identifier
            partial_text: What user has typed so far (can be empty for preemptive)
            scene_description: Visual scene description
            conversation_history: List of {speaker, text} dicts
            num_suggestions: Number of suggestions to return

        Returns:
            PredictionResult with suggestions and ghost text
        """
        start_time = time.time()

        logger.debug(
            "Generating predictions",
            partial_text=partial_text,
            has_scene=bool(scene_description),
            conv_turns=len(conversation_history) if conversation_history else 0,
        )

        if scene_description:
            self._cached_scene = scene_description

        # Get user's frequently used phrases
        user_phrases = []
        try:
            freq_data = await self._memory.get_user_phrase_frequencies(user_id, limit=10)
            user_phrases = [phrase for phrase, _ in freq_data[:5]]
        except Exception as e:
            logger.debug("Could not retrieve user phrases", error=str(e))

        # Load user profile context for personalization
        profile_context = ""
        try:
            from .user_profile import get_profile_service
            profile_service = get_profile_service()
            profile_context = await profile_service.get_profile_context_string(user_id)
        except Exception as e:
            logger.debug("Could not load user profile", error=str(e))

        # Try AI predictions via ModelManager (has automatic fallback)
        try:
            result = await self._generate_ai_predictions(
                partial_text=partial_text,
                scene_description=self._cached_scene or "Unknown environment",
                conversation_history=conversation_history or [],
                user_phrases=user_phrases,
                num_suggestions=num_suggestions,
                profile_context=profile_context,
                language=language,
            )
        except Exception as e:
            logger.error("All prediction models failed, using rule-based fallback", error=str(e))
            result = self._generate_fallback_predictions(
                partial_text=partial_text,
                conversation_history=conversation_history or [],
                language=language,
            )

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    async def _generate_ai_predictions(
        self,
        partial_text: str,
        scene_description: str,
        conversation_history: list[dict],
        user_phrases: list[str],
        num_suggestions: int,
        profile_context: str = "",
        language: str = "en",
    ) -> PredictionResult:
        """Generate predictions using LLM."""

        # Format conversation
        conv_str = ""
        if conversation_history:
            recent = conversation_history[-10:]
            conv_lines = []
            for turn in recent:
                speaker = turn.get("speaker", "other")
                text = turn.get("text", "")
                label = "User" if speaker == "user" else "Other person"
                conv_lines.append(f"- {label}: \"{text}\"")
            conv_str = f"""
RECENT CONVERSATION:
{chr(10).join(conv_lines)}

LAST THING SAID: "{conversation_history[-1].get('text', '')}" by {conversation_history[-1].get('speaker', 'other')}
"""

        # Determine prediction mode
        if not partial_text.strip():
            mode_instruction = """The user has NOT started typing yet. 
Suggest COMPLETE SENTENCES they might want to say based on the scene and conversation.
If someone just asked them a question, suggest answers.
If no conversation, suggest contextually appropriate openers."""
        else:
            mode_instruction = f"""The user has typed: "{partial_text}"
Suggest COMPLETIONS that continue from what they typed.
The first suggestion should complete their current word/thought.
Other suggestions can be alternative completions or full sentences starting with what they typed.
IMPORTANT: Each suggestion must logically continue from the typed text."""

        # Build profile section for prompt
        profile_section = f"\n{profile_context}\n" if profile_context else ""

        # Language instruction
        if language.startswith("ar"):
            lang_instruction = """IMPORTANT: The user is communicating in Arabic (العربية).
All suggestions, ghost_text, and completions MUST be in Arabic.
Write naturally in Arabic script. Do NOT transliterate or use English."""
        else:
            lang_instruction = "The user is communicating in English."

        prompt = f"""You are an AAC (Augmentative and Alternative Communication) predictive text assistant.
You help a person with speech/motor impairment communicate by predicting what they want to say.
{lang_instruction}
{profile_section}
VISUAL SCENE: {scene_description}
{conv_str}
USER'S FREQUENTLY USED PHRASES: {', '.join(user_phrases) if user_phrases else 'No history yet'}

{mode_instruction}

PRIORITY ORDER:
1. If a question was just asked to the user, prioritize ANSWERS to that question
2. Responses relevant to the ongoing conversation  
3. Statements relevant to the visual scene
4. The user's commonly used phrases (if contextually appropriate)
5. Responses matching the user's profile and communication style

Respond with a JSON object:
{{
  "ghost_text": "the single best inline completion or sentence suggestion (what would appear as faded text in the input field)",
  "suggestions": [
    {{"text": "suggested text", "confidence": 0.0-1.0, "is_completion": true/false}},
    ...
  ]
}}

- "ghost_text": The SINGLE best prediction. If user typed partial text, this completes it. If empty, this is the most likely sentence.
- "suggestions": {num_suggestions} alternative suggestions ranked by confidence.
- "is_completion": true if the suggestion continues/completes the partial text, false if it's a standalone sentence.

ONLY respond with valid JSON, no markdown or explanation."""

        try:
            # Use ModelManager which handles fallback + cooldown automatically
            response_text = await self._model_manager.generate(
                task_type="predictions",
                prompt=prompt,
                system_prompt="You are an AAC predictive text assistant. Respond ONLY with valid JSON.",
            )

            return self._parse_response(response_text, partial_text)

        except Exception as e:
            logger.error("AI prediction failed", error=str(e))
            return self._generate_fallback_predictions(partial_text, conversation_history)

    def _parse_response(self, response_text: str, partial_text: str) -> PredictionResult:
        """Parse LLM response into PredictionResult."""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            data = json.loads(text)

            ghost_text = data.get("ghost_text")
            suggestions_data = data.get("suggestions", [])

            suggestions = []
            for s in suggestions_data:
                suggestions.append(PredictedText(
                    text=s.get("text", ""),
                    confidence=float(s.get("confidence", 0.7)),
                    is_completion=bool(s.get("is_completion", False)),
                ))

            return PredictionResult(
                suggestions=suggestions,
                ghost_text=ghost_text,
            )

        except json.JSONDecodeError:
            logger.warning("Failed to parse prediction response")
            return self._generate_fallback_predictions(partial_text, [])

    def _generate_fallback_predictions(
        self,
        partial_text: str,
        conversation_history: list[dict],
        language: str = "en",
    ) -> PredictionResult:
        """Rule-based fallback when AI is unavailable."""

        # Check if last conversation turn was a question
        last_was_question = False
        if conversation_history:
            last_text = conversation_history[-1].get("text", "").strip().lower()
            last_was_question = (
                last_text.endswith("?") or
                last_text.startswith((
                    "do you", "are you", "can you", "will you", "would you",
                    "have you", "what", "where", "when", "who", "why", "how",
                ))
            )

        if not partial_text.strip():
            # Preemptive suggestions
            if last_was_question:
                suggestions = [
                    PredictedText("Yes", 0.95),
                    PredictedText("No", 0.95),
                    PredictedText("I don't know", 0.85),
                    PredictedText("Maybe", 0.8),
                    PredictedText("Yes, please", 0.75),
                ]
                ghost_text = "Yes"
            else:
                suggestions = [
                    PredictedText("I would like", 0.8),
                    PredictedText("Can you help me", 0.75),
                    PredictedText("Thank you", 0.7),
                    PredictedText("I need", 0.7),
                    PredictedText("Please", 0.65),
                ]
                ghost_text = "I would like"
        else:
            # Text-based completions
            pt = partial_text.lower().strip()
            completions = {
                "i": [("I would like", 0.9), ("I need", 0.85), ("I want", 0.8), ("I feel", 0.75), ("I am", 0.7)],
                "i w": [("I would like", 0.95), ("I want", 0.9), ("I was", 0.7), ("I will", 0.65), ("I wish", 0.6)],
                "i n": [("I need", 0.95), ("I need help", 0.85), ("I noticed", 0.6)],
                "th": [("Thank you", 0.9), ("That's great", 0.7), ("The", 0.65)],
                "he": [("Hello", 0.85), ("Help me", 0.8), ("Here", 0.6)],
                "pl": [("Please", 0.9), ("Please help", 0.8)],
                "ye": [("Yes", 0.95), ("Yes please", 0.85)],
                "no": [("No", 0.95), ("No thank you", 0.85), ("Not now", 0.7)],
                "ca": [("Can you", 0.9), ("Can I have", 0.8), ("Can you help me", 0.75)],
            }

            matched = []
            for prefix, comps in completions.items():
                if pt.startswith(prefix):
                    matched = comps
                    break

            if matched:
                suggestions = [PredictedText(t, c, is_completion=True) for t, c in matched]
                ghost_text = matched[0][0]
            else:
                suggestions = [
                    PredictedText(f"{partial_text} please", 0.6, is_completion=True),
                    PredictedText("Thank you", 0.5),
                    PredictedText("Yes", 0.5),
                ]
                ghost_text = f"{partial_text}..."

        return PredictionResult(suggestions=suggestions, ghost_text=ghost_text)

    async def store_accepted_suggestion(
        self,
        user_id: str,
        accepted_text: str,
        scene_description: str | None = None,
        conversation_context: str | None = None,
    ) -> None:
        """Store an accepted suggestion for learning."""
        try:
            from ..models import VisualContext
            visual_context = VisualContext(
                scene_description=scene_description or "Unknown",
                environmental_context="",
            )
            await self._memory.store_selection(
                user_id=user_id,
                selected_phrase=accepted_text,
                visual_context=visual_context,
            )
            logger.info(
                "Stored accepted prediction",
                user_id=user_id,
                text=accepted_text[:50],
            )
        except Exception as e:
            logger.error("Failed to store accepted prediction", error=str(e))

    async def format_text(self, raw_text: str) -> str:
        """
        AI-powered text formatting: fix capitalization, punctuation, spacing.
        Returns the cleaned-up text, or the original if formatting fails.
        """
        if not raw_text or not raw_text.strip():
            return raw_text

        prompt = (
            f'Fix the capitalization, punctuation, and spacing of this text. '
            f'Return ONLY the corrected text, nothing else.\n\n'
            f'Text: "{raw_text}"'
        )

        try:
            result = await self._model_manager.generate(
                task_type="formatting",
                prompt=prompt,
            )
            # Strip any quotes the model may have added
            formatted = result.strip().strip('"').strip("'").strip()
            if formatted:
                return formatted
            return raw_text
        except Exception as e:
            logger.warning("Text formatting failed, using raw text", error=str(e))
            return raw_text


# Singleton instance
_predictive_service: PredictiveSuggestionService | None = None


def get_predictive_suggestion_service() -> PredictiveSuggestionService:
    """Get the predictive suggestion service singleton."""
    global _predictive_service
    if _predictive_service is None:
        _predictive_service = PredictiveSuggestionService()
    return _predictive_service
