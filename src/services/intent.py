"""
SOWTEE Intent Prediction Service
Generates contextual phrase predictions using reasoning and memory.
"""

import time
import structlog
from groq import AsyncGroq

from ..config import get_settings
from ..models import MemoryRecord, PhraseCandidate, VisualContext
from .model_manager import log_model_input, log_model_output, log_model_error

logger = structlog.get_logger(__name__)


# Default phrase templates for common scenarios
DEFAULT_PHRASES: dict[str, list[str]] = {
    "drink": [
        ("I would like some water, please", "أريد بعض الماء، من فضلك"),
        ("Can I have something to drink?", "هل يمكنني الحصول على شيء للشرب؟"),
        ("I'm thirsty", "أنا عطشان"),
    ],
    "food": [
        ("I'm hungry", "أنا جائع"),
        ("Can I have something to eat?", "هل يمكنني الحصول على شيء للأكل؟"),
        ("I would like a snack", "أريد وجبة خفيفة"),
    ],
    "comfort": [
        ("I need to rest", "أحتاج للراحة"),
        ("Can you help me get comfortable?", "هل يمكنك مساعدتي لأكون مرتاحاً؟"),
        ("I'm feeling tired", "أشعر بالتعب"),
    ],
    "assistance": [
        ("I need help", "أحتاج مساعدة"),
        ("Can you come here, please?", "هل يمكنك المجيء هنا، من فضلك؟"),
        ("I need assistance", "أحتاج مساعدة"),
    ],
    "entertainment": [
        ("Can you turn on the TV?", "هل يمكنك تشغيل التلفزيون؟"),
        ("I'd like to watch something", "أود مشاهدة شيء ما"),
        ("Can you change the channel?", "هل يمكنك تغيير القناة؟"),
    ],
    "medical": [
        ("I'm in pain", "أشعر بألم"),
        ("I need my medication", "أحتاج دوائي"),
        ("Please call the nurse", "من فضلك اتصل بالممرضة"),
    ],
    "temperature": [
        ("I'm cold", "أنا بردان"),
        ("I'm hot", "أنا حرّان"),
        ("Can you adjust the temperature?", "هل يمكنك ضبط الحرارة؟"),
    ],
    "general": [
        ("Yes", "نعم"),
        ("No", "لا"),
        ("Thank you", "شكراً"),
    ],
}


class IntentPredictionService:
    """
    Intent prediction engine combining vision, memory, and LLM reasoning.
    
    Responsibilities:
    - Synthesize visual context and memories into intent predictions
    - Generate contextually appropriate phrases
    - Rank predictions by confidence and relevance
    - Provide bilingual phrase output (English/Arabic)
    """
    
    # Groq model for intent prediction
    INTENT_MODEL = "llama-3.3-70b-versatile"
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: AsyncGroq | None = None
        self._configure_model()
    
    def _configure_model(self) -> None:
        """Configure the reasoning model."""
        if not self.settings.groq_api_key:
            logger.warning("Groq API key not configured - using rule-based predictions")
            return
            
        self._client = AsyncGroq(api_key=self.settings.groq_api_key)
        logger.info("Groq intent client configured", model=self.INTENT_MODEL)
    
    async def predict_intent(
        self,
        visual_context: VisualContext,
        memories: list[MemoryRecord],
        additional_context: str = "",
    ) -> list[PhraseCandidate]:
        """
        Predict the user's communicative intent.
        
        This is the core of the REASON + ACT phases.
        
        Args:
            visual_context: Current visual scene analysis
            memories: Relevant memories from the user's history
            additional_context: Optional extra context from the user
            
        Returns:
            Top-k predicted phrases ranked by confidence
        """
        logger.debug("Predicting intent", objects_count=len(visual_context.detected_objects))
        
        # Strategy: Combine LLM reasoning with memory-based predictions
        candidates: list[PhraseCandidate] = []
        
        # 1. Memory-based predictions (highest confidence if recent)
        memory_candidates = self._generate_memory_predictions(memories, visual_context)
        candidates.extend(memory_candidates)
        
        # 2. LLM-based predictions
        if self._client:
            llm_candidates = await self._generate_llm_predictions(
                visual_context, memories, additional_context
            )
            candidates.extend(llm_candidates)
        else:
            # Rule-based fallback
            rule_candidates = self._generate_rule_predictions(visual_context)
            candidates.extend(rule_candidates)
        
        # Deduplicate and rank
        ranked = self._rank_candidates(candidates)
        
        return ranked[:self.settings.top_k_phrases]
    
    def _generate_memory_predictions(
        self,
        memories: list[MemoryRecord],
        visual_context: VisualContext
    ) -> list[PhraseCandidate]:
        """Generate predictions from user memory patterns."""
        candidates = []
        current_objects = {obj.label.lower() for obj in visual_context.detected_objects}
        
        for memory in memories[:5]:  # Top 5 most relevant memories
            # Boost confidence if objects match
            memory_objects = {obj.lower() for obj in memory.objects_present}
            overlap = len(current_objects & memory_objects)
            
            base_confidence = 0.7
            object_boost = min(overlap * 0.1, 0.2)
            frequency_boost = min(memory.selection_count * 0.02, 0.1)
            
            confidence = min(base_confidence + object_boost + frequency_boost, 0.95)
            
            candidates.append(PhraseCandidate(
                phrase=memory.selected_phrase,
                phrase_arabic=None,  # Would need translation service
                confidence=confidence,
                reasoning=f"Previously used in similar context ({memory.selection_count}x)",
                source="memory",
                related_objects=list(memory_objects & current_objects),
            ))
        
        return candidates
    
    async def _generate_llm_predictions(
        self,
        visual_context: VisualContext,
        memories: list[MemoryRecord],
        additional_context: str,
    ) -> list[PhraseCandidate]:
        """Generate predictions using LLM reasoning."""
        
        # Build context for LLM
        objects_list = ", ".join(obj.label for obj in visual_context.detected_objects)
        memory_phrases = [m.selected_phrase for m in memories[:3]]
        
        prompt = f"""You are an AAC (Augmentative and Alternative Communication) assistant helping a person with motor/speech impairment communicate.

VISUAL CONTEXT:
- Scene: {visual_context.scene_description}
- Objects visible: {objects_list}
- Location: {visual_context.environmental_context}
- Likely activity: {visual_context.activity_inference}

USER'S PREVIOUS PHRASES IN SIMILAR SITUATIONS:
{chr(10).join(f"- {p}" for p in memory_phrases) if memory_phrases else "No previous data"}

{f"ADDITIONAL CONTEXT: {additional_context}" if additional_context else ""}

Based on this context, predict the 3 most likely phrases the user might want to communicate.
Focus on:
1. Immediate needs related to visible objects
2. Comfort and assistance requests
3. Simple responses or acknowledgments

Respond with JSON array of exactly 3 predictions:
[
  {{"phrase": "English phrase", "phrase_arabic": "Arabic translation", "emotion": "[emotion]", "confidence": 0.0-1.0, "reasoning": "brief explanation"}},
  ...
]

For emotion, use one of: [cheerful], [friendly], [casual], [warm], [professionally], [whisper], [excited], [sad], [neutral].
Use [neutral] if no specific emotion is needed.

Only respond with valid JSON, no markdown or explanation."""

        try:
            start_time = time.time()
            
            # Log input
            log_model_input("intent", self.INTENT_MODEL, prompt)
            
            response = await self._client.chat.completions.create(
                model=self.INTENT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=512,
            )
            
            response_text = response.choices[0].message.content or ""
            
            # Log output
            duration_ms = (time.time() - start_time) * 1000
            log_model_output("intent", self.INTENT_MODEL, response_text, duration_ms)
            
            return self._parse_llm_predictions(response_text)
        except Exception as e:
            log_model_error("intent", self.INTENT_MODEL, str(e))
            logger.error("LLM prediction failed", error=str(e))
            return self._generate_rule_predictions(visual_context)
    
    def _parse_llm_predictions(self, response_text: str) -> list[PhraseCandidate]:
        """Parse LLM response into PhraseCandidate objects."""
        import json
        
        try:
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            
            predictions = json.loads(text)
            
            candidates = []
            for pred in predictions:
                candidates.append(PhraseCandidate(
                    phrase=pred.get("phrase", ""),
                    phrase_arabic=pred.get("phrase_arabic"),
                    confidence=float(pred.get("confidence", 0.5)),
                    reasoning=pred.get("reasoning", ""),
                    source="vision",
                    emotion=pred.get("emotion"),
                    related_objects=[],
                ))
            
            return candidates
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM predictions")
            return []
    
    def _generate_rule_predictions(
        self,
        visual_context: VisualContext
    ) -> list[PhraseCandidate]:
        """Generate predictions using rule-based matching."""
        candidates = []
        object_labels = {obj.label.lower() for obj in visual_context.detected_objects}
        
        # Match objects to phrase categories
        drink_words = {"water", "glass", "cup", "bottle", "drink", "beverage"}
        food_words = {"food", "plate", "fork", "spoon", "meal", "snack", "fruit"}
        entertainment_words = {"tv", "television", "remote", "screen", "phone"}
        comfort_words = {"bed", "pillow", "blanket", "chair", "couch"}
        
        matched_categories = []
        
        if object_labels & drink_words:
            matched_categories.append("drink")
        if object_labels & food_words:
            matched_categories.append("food")
        if object_labels & entertainment_words:
            matched_categories.append("entertainment")
        if object_labels & comfort_words:
            matched_categories.append("comfort")
        
        # Add general if no matches
        if not matched_categories:
            matched_categories = ["general", "assistance"]
        
        for category in matched_categories[:2]:
            phrases = DEFAULT_PHRASES.get(category, DEFAULT_PHRASES["general"])
            for i, (phrase_en, phrase_ar) in enumerate(phrases[:2]):
                confidence = 0.6 - (i * 0.1)
                # Assign emotion based on category
                emotion = "[neutral]"
                if category in ["drink", "food", "comfort"]:
                     emotion = "[friendly]"
                elif category == "medical":
                     emotion = "[whisper]"
                elif category == "entertainment":
                     emotion = "[excited]"
                
                candidates.append(PhraseCandidate(
                    phrase=phrase_en,
                    phrase_arabic=phrase_ar,
                    confidence=confidence,
                    reasoning=f"Matched category: {category}",
                    source="rules",
                    emotion=emotion,
                    related_objects=list(object_labels)[:3],
                ))
        
        return candidates
    
    def _rank_candidates(
        self,
        candidates: list[PhraseCandidate]
    ) -> list[PhraseCandidate]:
        """Deduplicate and rank phrase candidates."""
        # Deduplicate by phrase text
        seen_phrases: dict[str, PhraseCandidate] = {}
        
        for candidate in candidates:
            phrase_key = candidate.phrase.lower().strip()
            if phrase_key not in seen_phrases:
                seen_phrases[phrase_key] = candidate
            else:
                # Keep the one with higher confidence
                existing = seen_phrases[phrase_key]
                if candidate.confidence > existing.confidence:
                    seen_phrases[phrase_key] = candidate
        
        # Sort by confidence
        ranked = sorted(
            seen_phrases.values(),
            key=lambda x: x.confidence,
            reverse=True
        )
        
        return ranked


# Singleton instance
_intent_service: IntentPredictionService | None = None


def get_intent_service() -> IntentPredictionService:
    """Get the intent prediction service singleton."""
    global _intent_service
    if _intent_service is None:
        _intent_service = IntentPredictionService()
    return _intent_service
