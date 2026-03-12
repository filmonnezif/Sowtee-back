"""
ElevenLabs TTS Service
Uses ElevenLabs API for high-quality, expressive text-to-speech
with voice-settings-based emotion control.
"""

import os
from typing import AsyncGenerator

import structlog

logger = structlog.get_logger(__name__)

# Chunk size for streaming audio bytes
STREAM_CHUNK_SIZE = 8192  # 8KB


async def generate_elevenlabs_speech(
    text: str,
    language: str = "en",
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.0,
    speed: float = 1.0,
) -> AsyncGenerator[bytes, None]:
    """
    Generate speech using ElevenLabs API with emotion-aware voice settings.

    Args:
        text: Text to convert to speech.
        language: Language code ("en", "ar", etc.) — the multilingual model
                  handles language automatically from the text content.
        stability: 0.0–1.0. Lower = more emotional range, higher = more stable/monotone.
        similarity_boost: 0.0–1.0. How closely to match the original voice.
        style: 0.0–1.0. Amplifies speaker expressiveness/mannerisms.
        speed: Playback speed multiplier (1.0 = normal).

    Yields:
        Audio bytes chunks (MP3 format).
    """
    from ..config import get_settings
    from .error_logger import get_error_logger
    from .model_manager import log_model_input, log_model_output, log_model_error

    error_logger = get_error_logger()
    settings = get_settings()

    api_key = settings.elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY", "")
    voice_id = settings.elevenlabs_voice_id or os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
    model_id = settings.elevenlabs_model_id or os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")

    # Use cloned voice if available
    from .voice_clone import get_cloned_voice_id
    cloned_id = get_cloned_voice_id()
    if cloned_id:
        voice_id = cloned_id
        logger.info("using_cloned_voice", voice_id=voice_id)

    if not api_key:
        logger.error("ELEVENLABS_API_KEY not set")
        error_logger.log_error(
            source="elevenlabs_tts",
            error_type="config_error",
            message="ELEVENLABS_API_KEY not configured",
            details={"text_length": len(text)},
        )
        raise ValueError("ELEVENLABS_API_KEY not configured")

    log_model_input(
        "elevenlabs_tts",
        model_id,
        text,
        f"voice={voice_id} stability={stability} similarity_boost={similarity_boost} style={style} speed={speed}",
    )

    import time
    start = time.time()

    try:
        from elevenlabs import VoiceSettings
        from elevenlabs.client import AsyncElevenLabs

        client = AsyncElevenLabs(api_key=api_key)

        # Generate speech with voice settings for emotion control
        audio_iter = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                speed=speed,
                use_speaker_boost=True,
            ),
            output_format="mp3_44100_128",
        )

        # Stream the audio bytes
        total_bytes = 0
        async for chunk in audio_iter:
            if chunk:
                total_bytes += len(chunk)
                yield chunk

        duration_ms = (time.time() - start) * 1000
        log_model_output(
            "elevenlabs_tts",
            model_id,
            f"Generated {total_bytes} bytes of audio",
            duration_ms,
        )

        logger.info(
            "ElevenLabs TTS generation successful",
            text_length=len(text),
            audio_bytes=total_bytes,
            duration_ms=round(duration_ms),
            model=model_id,
            voice=voice_id,
        )

    except Exception as e:
        log_model_error("elevenlabs_tts", model_id, str(e))
        error_logger.log_error(
            source="elevenlabs_tts",
            error_type="tts_error",
            message=f"ElevenLabs TTS failed: {type(e).__name__}",
            details={"error": str(e), "text_length": len(text)},
        )
        raise
