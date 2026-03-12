"""
SOWTEE Word Suggestion Service
Provides intelligent, context-aware word and phrase suggestions for sentence building.
Uses Groq AI for understanding context and generating relevant suggestions.
"""

import structlog
import time
from uuid import uuid4

from groq import AsyncGroq

from ..config import get_settings
from .model_manager import log_model_input, log_model_output, log_model_error
from ..models import (
    WordSuggestion,
    WordSuggestionRequest,
    WordSuggestionResponse,
    WordCategory,
    VisualContext,
)
from .memory import get_memory_service
from .vision import get_vision_service

logger = structlog.get_logger(__name__)


# Context-aware word suggestions organized by sentence position and scene type
CONTEXTUAL_WORDS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "kitchen": {
        "starters": [("I want", "أريد"), ("Can I have", "هل يمكنني الحصول على"), ("Please give me", "من فضلك أعطني"), ("I need", "أحتاج")],
        "objects": [("water", "ماء"), ("food", "طعام"), ("coffee", "قهوة"), ("tea", "شاي"), ("snack", "وجبة خفيفة"), ("medicine", "دواء")],
        "verbs": [("eat", "آكل"), ("drink", "أشرب"), ("have", "أملك"), ("get", "أحصل على")],
        "phrases": [
            ("I would like some water", "أريد بعض الماء"),
            ("Can I have something to eat?", "هل يمكنني الحصول على شيء للأكل؟"),
            ("I'm hungry", "أنا جائع"),
            ("I'm thirsty", "أنا عطشان"),
        ],
    },
    "living_room": {
        "starters": [("Can you", "هل يمكنك"), ("Please", "من فضلك"), ("I want to", "أريد أن"), ("Turn on", "شغّل")],
        "objects": [("TV", "التلفزيون"), ("remote", "جهاز التحكم"), ("blanket", "بطانية"), ("light", "ضوء")],
        "verbs": [("watch", "أشاهد"), ("turn on", "شغّل"), ("turn off", "أطفئ"), ("change", "غيّر")],
        "phrases": [
            ("Can you turn on the TV?", "هل يمكنك تشغيل التلفزيون؟"),
            ("I want to watch something", "أريد أن أشاهد شيئاً"),
            ("Please change the channel", "من فضلك غيّر القناة"),
            ("I need the remote", "أحتاج جهاز التحكم"),
        ],
    },
    "bedroom": {
        "starters": [("I need", "أحتاج"), ("Help me", "ساعدني"), ("I feel", "أشعر"), ("Can you", "هل يمكنك")],
        "objects": [("pillow", "وسادة"), ("blanket", "بطانية"), ("light", "ضوء"), ("bed", "سرير")],
        "verbs": [("sleep", "أنام"), ("rest", "أرتاح"), ("lie down", "أستلقي")],
        "phrases": [
            ("I need to rest", "أحتاج للراحة"),
            ("I'm feeling tired", "أشعر بالتعب"),
            ("Please turn off the light", "من فضلك أطفئ الضوء"),
            ("Help me get comfortable", "ساعدني لأكون مرتاحاً"),
        ],
    },
    "bathroom": {
        "starters": [("I need to", "أحتاج أن"), ("Help me", "ساعدني"), ("Can you", "هل يمكنك")],
        "objects": [("bathroom", "حمام"), ("water", "ماء"), ("towel", "منشفة")],
        "verbs": [("go", "أذهب"), ("use", "أستخدم")],
        "phrases": [
            ("I need to use the bathroom", "أحتاج لاستخدام الحمام"),
            ("Help me to the bathroom", "ساعدني للذهاب للحمام"),
            ("I need some water", "أحتاج بعض الماء"),
        ],
    },
    "general": {
        "starters": [("I", "أنا"), ("Can", "هل يمكن"), ("Please", "من فضلك"), ("Help", "ساعد"), ("I want", "أريد"), ("I need", "أحتاج")],
        "subjects": [("I", "أنا"), ("you", "أنت"), ("we", "نحن"), ("someone", "شخص ما")],
        "verbs": [("want", "أريد"), ("need", "أحتاج"), ("feel", "أشعر"), ("have", "أملك"), ("see", "أرى"), ("help", "ساعد")],
        "objects": [("help", "مساعدة"), ("water", "ماء"), ("food", "طعام"), ("rest", "راحة"), ("medicine", "دواء"), ("phone", "هاتف")],
        "modifiers": [("now", "الآن"), ("please", "من فضلك"), ("thanks", "شكراً"), ("soon", "قريباً"), ("more", "أكثر")],
        "phrases": [
            ("I need help", "أحتاج مساعدة"),
            ("Yes", "نعم"),
            ("No", "لا"),
            ("Thank you", "شكراً"),
            ("I'm in pain", "أشعر بألم"),
            ("Please call someone", "من فضلك اتصل بشخص ما"),
        ],
    },
}


class WordSuggestionService:
    """
    Intelligent word suggestion service for AAC sentence building.
    
    Uses Groq AI to understand context and provide relevant suggestions
    based on:
    - Current visual scene (what the user sees)
    - Sentence being built (what they've already selected)
    - User's history and preferences (from memory)
    - Natural language patterns
    """
    
    # Groq model for word suggestions
    SUGGESTION_MODEL = "llama-3.3-70b-versatile"
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self._vision = get_vision_service()
        self._memory = get_memory_service()
        self._client: AsyncGroq | None = None
        self._configure_model()
        
        # Cache for scene context
        self._cached_scene: str | None = None
        self._cached_env: str | None = None
    
    def _configure_model(self) -> None:
        """Configure the Groq client for word suggestions."""
        if not self.settings.groq_api_key:
            logger.warning("Groq API key not configured - using rule-based suggestions")
            return
            
        self._client = AsyncGroq(api_key=self.settings.groq_api_key)
        logger.info("Groq client configured for suggestions", model=self.SUGGESTION_MODEL)
    
    async def get_suggestions(self, request: WordSuggestionRequest) -> WordSuggestionResponse:
        """
        Get intelligent word and phrase suggestions for sentence building.
        
        This is the main entry point that:
        1. Analyzes the scene if an image is provided
        2. Considers the current sentence context
        3. Retrieves relevant user memories
        4. Generates contextually appropriate suggestions
        """
        start_time = time.time()
        session_id = str(uuid4())
        
        logger.debug(
            "Generating word suggestions",
            current_sentence=request.current_sentence,
            has_image=bool(request.image_base64),
            conversation_context_count=len(request.conversation_context) if request.conversation_context else 0,
            last_heard=request.conversation_context[-1] if request.conversation_context else None,
        )
        
        # Get scene context
        scene_description = request.scene_context
        env_context = "general"
        
        if request.image_base64:
            try:
                visual_context = await self._vision.analyze_scene(request.image_base64)
                scene_description = visual_context.scene_description
                env_context = self._categorize_environment(visual_context.environmental_context)
                self._cached_scene = scene_description
                self._cached_env = env_context
            except Exception as e:
                logger.warning("Scene analysis failed, using cached/general context", error=str(e))
                scene_description = self._cached_scene or "General indoor environment"
                env_context = self._cached_env or "general"
        
        # Get suggestions using Gemini if available
        if self._client:
            suggestions = await self._generate_ai_suggestions(
                request.current_sentence,
                scene_description or "Unknown scene",
                env_context,
                request.user_id,
                request.conversation_context,
            )
        else:
            suggestions = self._generate_rule_based_suggestions(
                request.current_sentence,
                env_context,
                request.conversation_context,
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return WordSuggestionResponse(
            session_id=session_id,
            words=suggestions,
            scene_description=scene_description,
            processing_time_ms=processing_time,
        )
    
    def _categorize_environment(self, env_string: str) -> str:
        """Categorize environment into known categories."""
        env_lower = env_string.lower()
        
        if any(word in env_lower for word in ["kitchen", "cooking", "food prep"]):
            return "kitchen"
        elif any(word in env_lower for word in ["living", "lounge", "tv", "couch"]):
            return "living_room"
        elif any(word in env_lower for word in ["bed", "sleep", "rest"]):
            return "bedroom"
        elif any(word in env_lower for word in ["bath", "toilet", "shower"]):
            return "bathroom"
        else:
            return "general"
    
    async def _generate_ai_suggestions(
        self,
        current_sentence: list[str],
        scene_description: str,
        env_context: str,
        user_id: str,
        conversation_context: list[str] | None = None,
    ) -> list[WordSuggestion]:
        """Generate suggestions using Gemini AI."""
        
        # Get user's frequently used phrases
        user_phrases = []
        try:
            freq_data = await self._memory.get_user_phrase_frequencies(user_id, limit=10)
            user_phrases = [phrase for phrase, _ in freq_data[:5]]
        except Exception as e:
            logger.debug("Could not retrieve user phrases", error=str(e))
        
        sentence_so_far = " ".join(current_sentence) if current_sentence else "(empty - starting new sentence)"
        
        # Format conversation context with emphasis
        conv_context_str = ""
        last_heard_is_question = False
        if conversation_context and len(conversation_context) > 0:
            recent_speech = conversation_context[-10:]  # Last 10 entries
            conv_context_str = f"""
RECENT CONVERSATION (what was heard around the user - THIS IS THE MOST IMPORTANT CONTEXT):
{chr(10).join(f'- "{speech}"' for speech in recent_speech)}

MOST RECENT THING HEARD: "{conversation_context[-1]}"
"""
            # Detect if the last thing heard was a question
            last_heard = conversation_context[-1].strip().lower()
            last_heard_is_question = (
                last_heard.endswith('?') or
                last_heard.startswith(('do you', 'are you', 'can you', 'will you', 'would you', 'could you',
                                       'have you', 'is it', 'is there', 'what', 'where', 'when', 'who', 
                                       'why', 'how', 'which', 'shall', 'should'))
            )
        
        # Build priority context for the prompt
        question_guidance = ""
        if last_heard_is_question:
            question_guidance = """
⚠️ IMPORTANT: A QUESTION WAS JUST ASKED TO THE USER!
Prioritize suggesting DIRECT ANSWERS to this question:
- "Yes", "No", "Maybe", "I don't know"
- Specific answers relevant to the question
- Short confirming/denying responses
The user needs to RESPOND to what was just asked.
"""
        
        prompt = f"""You are an AAC (Augmentative and Alternative Communication) assistant helping a person with motor/speech impairment build sentences word by word.

{question_guidance}
CURRENT SCENE: {scene_description}
ENVIRONMENT: {env_context}
SENTENCE SO FAR: {sentence_so_far}
{conv_context_str}
USER'S COMMON PHRASES (lower priority): {', '.join(user_phrases) if user_phrases else 'No history yet'}

CRITICAL PRIORITY ORDER for suggestions:
1. FIRST: If a question was just asked, suggest direct answers (Yes/No/specific answers)  
2. SECOND: Words/phrases that respond to or continue the recent conversation
3. THIRD: Words relevant to the current visual scene
4. FOURTH: User's common phrases (only if they fit the current context)

The goal is to help the user carry on a NATURAL CONVERSATION. Think about what someone would naturally want to say in response to what they just heard.

Respond with a JSON array of exactly 8 suggestions:
[
  {{"word": "word or phrase", "word_arabic": "Arabic translation", "confidence": 0.0-1.0, "category": "starter|verb|object|modifier|ending|phrase|answer", "related_to_scene": true/false}},
  ...
]

Categories:
- "answer": Direct responses (Yes, No, Maybe, OK, specific answers) - USE THIS FOR QUESTION RESPONSES
- "starter": Sentence beginnings (I, Can, Please, Help)
- "verb": Action words (want, need, have, go)
- "object": Things/nouns (water, food, help, medicine)
- "modifier": Descriptors and endings (now, please, more)
- "ending": Sentence completions (., ?, thanks)
- "phrase": Complete phrase suggestions

ONLY respond with valid JSON, no markdown or explanation."""

        try:
            start_time = time.time()
            
            # Log input
            log_model_input("suggestions", self.SUGGESTION_MODEL, prompt)
            
            response = await self._client.chat.completions.create(
                model=self.SUGGESTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024,
            )
            
            response_text = response.choices[0].message.content or ""
            
            # Log output
            duration_ms = (time.time() - start_time) * 1000
            log_model_output("suggestions", self.SUGGESTION_MODEL, response_text, duration_ms)
            
            suggestions = self._parse_ai_response(response_text)
            
            # Ensure we have enough suggestions
            if len(suggestions) < 4:
                fallback = self._generate_rule_based_suggestions(current_sentence, env_context, conversation_context)
                suggestions.extend(fallback[:8 - len(suggestions)])
            
            return suggestions[:8]
            
        except Exception as e:
            log_model_error("suggestions", self.SUGGESTION_MODEL, str(e))
            logger.error("AI suggestion generation failed", error=str(e))
            return self._generate_rule_based_suggestions(current_sentence, env_context, conversation_context)
    
    def _parse_ai_response(self, response_text: str) -> list[WordSuggestion]:
        """Parse Gemini's JSON response into WordSuggestion objects."""
        import json
        
        try:
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            
            predictions = json.loads(text)
            
            suggestions = []
            for pred in predictions:
                category_str = pred.get("category", "object").lower()
                
                # Map category string to enum
                category_map = {
                    "answer": WordCategory.STARTER,  # Direct answers start responses
                    "starter": WordCategory.STARTER,
                    "subject": WordCategory.SUBJECT,
                    "verb": WordCategory.VERB,
                    "object": WordCategory.OBJECT,
                    "modifier": WordCategory.MODIFIER,
                    "ending": WordCategory.ENDING,
                    "phrase": WordCategory.STARTER,  # Treat phrases as starters for display
                }
                category = category_map.get(category_str, WordCategory.OBJECT)
                
                suggestions.append(WordSuggestion(
                    word=pred.get("word", ""),
                    word_arabic=pred.get("word_arabic"),
                    confidence=float(pred.get("confidence", 0.7)),
                    category=category,
                    related_to_scene=bool(pred.get("related_to_scene", False)),
                ))
            
            return suggestions
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI suggestions")
            return []
    
    def _generate_rule_based_suggestions(
        self,
        current_sentence: list[str],
        env_context: str,
        conversation_context: list[str] | None = None,
    ) -> list[WordSuggestion]:
        """Generate suggestions using rule-based matching when AI is unavailable."""
        
        context_words = CONTEXTUAL_WORDS.get(env_context, CONTEXTUAL_WORDS["general"])
        general_words = CONTEXTUAL_WORDS["general"]
        
        suggestions = []
        sentence_length = len(current_sentence)
        
        # Check if a question was asked - prioritize direct answers
        last_heard_is_question = False
        if conversation_context and len(conversation_context) > 0:
            last_heard = conversation_context[-1].strip().lower()
            last_heard_is_question = (
                last_heard.endswith('?') or
                last_heard.startswith(('do you', 'are you', 'can you', 'will you', 'would you', 'could you',
                                       'have you', 'is it', 'is there', 'what', 'where', 'when', 'who', 
                                       'why', 'how', 'which', 'shall', 'should'))
            )
        
        # If a question was asked and sentence is empty, prioritize direct answers
        if last_heard_is_question and sentence_length == 0:
            # Add direct answer options first
            direct_answers = [
                ("Yes", "نعم"),
                ("No", "لا"),
                ("Maybe", "ربما"),
                ("I don't know", "لا أعرف"),
                ("Yes, please", "نعم، من فضلك"),
                ("No, thank you", "لا، شكراً"),
            ]
            for word, arabic in direct_answers:
                suggestions.append(WordSuggestion(
                    word=word,
                    word_arabic=arabic,
                    confidence=0.95,
                    category=WordCategory.STARTER,
                    related_to_scene=False,
                ))
            
            # Add a couple more contextual starters
            for word, arabic in context_words.get("starters", general_words["starters"])[:2]:
                suggestions.append(WordSuggestion(
                    word=word,
                    word_arabic=arabic,
                    confidence=0.8,
                    category=WordCategory.STARTER,
                    related_to_scene=True,
                ))
                
        elif sentence_length == 0:
            # Empty sentence - suggest starters and phrases
            for word, arabic in context_words.get("starters", general_words["starters"])[:3]:
                suggestions.append(WordSuggestion(
                    word=word,
                    word_arabic=arabic,
                    confidence=0.9,
                    category=WordCategory.STARTER,
                    related_to_scene=True,
                ))
            
            # Add phrase suggestions
            for phrase, arabic in context_words.get("phrases", general_words["phrases"])[:4]:
                suggestions.append(WordSuggestion(
                    word=phrase,
                    word_arabic=arabic,
                    confidence=0.85,
                    category=WordCategory.STARTER,
                    related_to_scene=True,
                ))
                
        elif sentence_length == 1:
            # After starter - suggest verbs
            for word, arabic in context_words.get("verbs", general_words["verbs"])[:4]:
                suggestions.append(WordSuggestion(
                    word=word,
                    word_arabic=arabic,
                    confidence=0.85,
                    category=WordCategory.VERB,
                    related_to_scene=True,
                ))
            
            # Add some objects too
            for word, arabic in context_words.get("objects", general_words["objects"])[:3]:
                suggestions.append(WordSuggestion(
                    word=word,
                    word_arabic=arabic,
                    confidence=0.75,
                    category=WordCategory.OBJECT,
                    related_to_scene=True,
                ))
                
        elif sentence_length == 2:
            # After verb - suggest objects
            for word, arabic in context_words.get("objects", general_words["objects"])[:5]:
                suggestions.append(WordSuggestion(
                    word=word,
                    word_arabic=arabic,
                    confidence=0.85,
                    category=WordCategory.OBJECT,
                    related_to_scene=True,
                ))
            
            # Add modifiers
            for word, arabic in general_words.get("modifiers", [])[:3]:
                suggestions.append(WordSuggestion(
                    word=word,
                    word_arabic=arabic,
                    confidence=0.7,
                    category=WordCategory.MODIFIER,
                    related_to_scene=False,
                ))
        else:
            # Later in sentence - suggest modifiers and endings
            for word, arabic in general_words.get("modifiers", [])[:5]:
                suggestions.append(WordSuggestion(
                    word=word,
                    word_arabic=arabic,
                    confidence=0.8,
                    category=WordCategory.MODIFIER,
                    related_to_scene=False,
                ))
            
            # Add some objects that might complete the thought
            for word, arabic in context_words.get("objects", general_words["objects"])[:3]:
                suggestions.append(WordSuggestion(
                    word=word,
                    word_arabic=arabic,
                    confidence=0.7,
                    category=WordCategory.OBJECT,
                    related_to_scene=True,
                ))
        
        return suggestions[:8]


# Singleton instance
_word_suggestion_service: WordSuggestionService | None = None


def get_word_suggestion_service() -> WordSuggestionService:
    """Get the word suggestion service singleton."""
    global _word_suggestion_service
    if _word_suggestion_service is None:
        _word_suggestion_service = WordSuggestionService()
    return _word_suggestion_service
