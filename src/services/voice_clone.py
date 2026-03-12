"""
Voice Cloning Service — ElevenLabs Instant Voice Cloning (IVC)
Allows users to upload a short audio sample to clone their voice.
"""

import structlog
from src.config import get_settings

logger = structlog.get_logger(__name__)

# Module-level storage for cloned voice ID
_cloned_voice_id: str | None = None
_cloned_voice_name: str | None = None


def get_cloned_voice_id() -> str | None:
    """Return the active cloned voice ID, or None if not set."""
    return _cloned_voice_id


def get_clone_status() -> dict:
    """Return current clone status."""
    return {
        "is_cloned": _cloned_voice_id is not None,
        "voice_id": _cloned_voice_id,
        "voice_name": _cloned_voice_name,
    }


def clear_cloned_voice() -> dict:
    """Reset to default voice."""
    global _cloned_voice_id, _cloned_voice_name
    old_id = _cloned_voice_id
    _cloned_voice_id = None
    _cloned_voice_name = None
    logger.info("voice_clone_cleared", old_voice_id=old_id)
    return {"status": "cleared", "old_voice_id": old_id}


async def clone_voice(audio_bytes: bytes, filename: str, voice_name: str = "My Voice") -> dict:
    """
    Clone a voice from an audio file using ElevenLabs IVC.

    Args:
        audio_bytes: Raw audio file bytes
        filename: Original filename (for content type detection)
        voice_name: Name for the cloned voice

    Returns:
        dict with voice_id and status
    """
    global _cloned_voice_id, _cloned_voice_name

    settings = get_settings()
    api_key = settings.elevenlabs_api_key

    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not configured")

    logger.info(
        "voice_clone_starting",
        filename=filename,
        voice_name=voice_name,
        file_size=len(audio_bytes),
    )

    try:
        from elevenlabs.client import AsyncElevenLabs

        client = AsyncElevenLabs(api_key=api_key)

        # Create voice clone via IVC (Instant Voice Cloning)
        # Note: In elevenlabs>=1.0.0, use client.voices.add()
        result = await client.voices.add(
            name=voice_name,
            files=[audio_bytes],
            description=f"Cloned voice from {filename}",
        )

        _cloned_voice_id = result.voice_id
        _cloned_voice_name = voice_name

        logger.info(
            "voice_clone_success",
            voice_id=_cloned_voice_id,
            voice_name=voice_name,
        )

        return {
            "status": "success",
            "voice_id": _cloned_voice_id,
            "voice_name": voice_name,
        }

    except Exception as e:
        logger.error(
            "voice_clone_failed",
            error=str(e),
            error_type=type(e).__name__,
            filename=filename,
        )
        raise
