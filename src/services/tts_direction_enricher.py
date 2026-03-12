"""
TTS Direction Enricher — ElevenLabs Voice Settings Edition
Converts text emotion/context into ElevenLabs VoiceSettings parameters
(stability, similarity_boost, style, speed) instead of Orpheus bracket tags.

Uses Groq's llama-3.3-70b-versatile for fast inference of emotion analysis.
"""

import json
import re
import structlog

logger = structlog.get_logger(__name__)

# Fast model for emotion inference
DIRECTION_MODEL = "llama-3.3-70b-versatile"

# Default voice settings — cheerful & calm baseline
DEFAULT_VOICE_SETTINGS = {
    "stability": 0.65,
    "similarity_boost": 0.80,
    "style": 0.20,
    "speed": 0.85,
}

SYSTEM_PROMPT = """\
You are an emotion analyzer for a text-to-speech system used by people with speech disabilities (ALS/MS).
The voice should always sound WARM, CHEERFUL, and NATURAL — like a kind friend speaking calmly.
Never produce settings that sound robotic, rushed, or flat.

You MUST return a JSON object with these exact keys (all floats):
- "stability": 0.0–1.0. Higher = calmer, smoother voice. ALWAYS use 0.55+ to avoid robotic jitter.
  Use 0.55–0.65 for emotional/urgent text, 0.65–0.75 for normal cheerful, 0.80+ for very calm/steady.
- "similarity_boost": 0.0–1.0. How closely to match the voice. Always use 0.75–0.85.
- "style": 0.0–1.0. Adds warmth and expressiveness. Use 0.15–0.30 for most speech. Only go above 0.4 for very dramatic moments.
- "speed": 0.7–1.0. IMPORTANT: Keep speed at 0.82–0.90 for most text. Never go above 0.95. Slower = more natural and human.

Guidelines (aim for warm & cheerful baseline):
- Urgent/help requests: stability 0.55, style 0.30, speed 0.85
- Greetings/cheerful: stability 0.65, style 0.25, speed 0.88
- Pain/discomfort: stability 0.60, style 0.15, speed 0.80
- Gratitude/warmth: stability 0.70, style 0.25, speed 0.85
- Neutral/factual: stability 0.70, style 0.15, speed 0.85
- Humor/playful: stability 0.60, style 0.35, speed 0.88
- Sad/emotional: stability 0.60, style 0.20, speed 0.80

Output ONLY the JSON object. No explanations, no markdown, no extra text.

Example:
Input: "I need water"
Context: Medical care setting
{"stability": 0.55, "similarity_boost": 0.80, "style": 0.30, "speed": 0.85}

Input: "Good morning! How are you today?"
Context: Social gathering
{"stability": 0.65, "similarity_boost": 0.80, "style": 0.25, "speed": 0.88}

Input: "Thank you for helping me"
Context: Caregiver interaction
{"stability": 0.70, "similarity_boost": 0.80, "style": 0.25, "speed": 0.85}"""


async def get_voice_settings(
    text: str,
    scene_description: str | None = None,
    conversation_context: str | None = None,
    custom_context: str | None = None,
) -> dict:
    """
    Analyze text emotion and return ElevenLabs voice settings.

    Args:
        text: The sentence to be spoken.
        scene_description: Visual scene context.
        conversation_context: Recent conversation history.
        custom_context: User-defined situation.

    Returns:
        Dict with keys: stability, similarity_boost, style, speed
    """
    from .model_manager import get_model_manager, log_model_input, log_model_output, log_model_error

    manager = get_model_manager()
    client = manager._get_client()

    # Build context string
    context_parts = []
    if custom_context:
        context_parts.append(f"Situation: {custom_context}")
    if scene_description:
        context_parts.append(f"Scene: {scene_description}")
    if conversation_context:
        context_parts.append(f"Recent conversation: {conversation_context}")

    context_str = "; ".join(context_parts) if context_parts else "General conversation"

    user_prompt = f'Input: "{text}"\nContext: {context_str}'

    log_model_input("voice_settings", DIRECTION_MODEL, user_prompt, SYSTEM_PROMPT)

    import time
    start = time.time()

    try:
        response = await client.chat.completions.create(
            model=DIRECTION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=128,
        )

        result = response.choices[0].message.content or ""
        result = result.strip()

        duration_ms = (time.time() - start) * 1000
        log_model_output("voice_settings", DIRECTION_MODEL, result, duration_ms)

        # Parse JSON response
        settings = _parse_voice_settings(result)
        logger.info("Voice settings determined", text_preview=text[:50], settings=settings)
        return settings

    except Exception as e:
        log_model_error("voice_settings", DIRECTION_MODEL, str(e))
        logger.warning("Voice settings inference failed, using fallback", error=str(e))
        return _fallback_settings(text)


def _parse_voice_settings(raw: str) -> dict:
    """Parse the LLM JSON response into validated voice settings."""
    # Try to extract JSON from the response (handle markdown code blocks)
    json_match = re.search(r'\{[^}]+\}', raw)
    if not json_match:
        logger.warning("No JSON found in voice settings response", raw=raw)
        return DEFAULT_VOICE_SETTINGS.copy()

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in voice settings response", raw=raw)
        return DEFAULT_VOICE_SETTINGS.copy()

    # Validate and clamp values
    settings = {}
    for key, default in DEFAULT_VOICE_SETTINGS.items():
        val = parsed.get(key, default)
        if not isinstance(val, (int, float)):
            val = default
        # Clamp to valid ranges
        if key == "speed":
            val = max(0.7, min(1.3, float(val)))
        else:
            val = max(0.0, min(1.0, float(val)))
        settings[key] = round(val, 2)

    return settings


def _fallback_settings(text: str) -> dict:
    """Simple keyword-based fallback when LLM is unavailable."""
    lower = text.lower()

    if any(w in lower for w in ("help", "emergency", "urgent", "pain", "hurt", "nurse", "doctor")):
        return {"stability": 0.55, "similarity_boost": 0.80, "style": 0.30, "speed": 0.85}
    if any(w in lower for w in ("thank", "grateful", "appreciate")):
        return {"stability": 0.70, "similarity_boost": 0.80, "style": 0.25, "speed": 0.85}
    if any(w in lower for w in ("hello", "hi ", "good morning", "good afternoon", "good evening", "hey")):
        return {"stability": 0.65, "similarity_boost": 0.80, "style": 0.25, "speed": 0.88}
    if any(w in lower for w in ("sorry", "apologize", "forgive")):
        return {"stability": 0.65, "similarity_boost": 0.80, "style": 0.20, "speed": 0.82}
    if any(w in lower for w in ("happy", "great", "wonderful", "amazing", "love")):
        return {"stability": 0.60, "similarity_boost": 0.80, "style": 0.30, "speed": 0.88}
    if any(w in lower for w in ("sad", "miss", "lonely", "tired")):
        return {"stability": 0.60, "similarity_boost": 0.80, "style": 0.20, "speed": 0.80}
    if any(w in lower for w in ("please", "could you", "can you", "would you")):
        return {"stability": 0.65, "similarity_boost": 0.80, "style": 0.20, "speed": 0.85}
    if "?" in text:
        return {"stability": 0.65, "similarity_boost": 0.80, "style": 0.20, "speed": 0.85}
    if "!" in text:
        return {"stability": 0.60, "similarity_boost": 0.80, "style": 0.30, "speed": 0.88}

    return DEFAULT_VOICE_SETTINGS.copy()
