"""
Voice Cloning Service — ElevenLabs Instant Voice Cloning (IVC)
Allows users to upload a short audio sample to clone their voice.
"""

import structlog
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.config import get_settings

logger = structlog.get_logger(__name__)

# Module-level storage for cloned voice ID
_cloned_voice_id: str | None = None
_cloned_voice_name: str | None = None


def _extract_voice_id(result: object) -> str | None:
    """Extract voice_id across SDK response shapes."""
    if isinstance(result, dict):
        voice_id = result.get("voice_id")
        return voice_id if isinstance(voice_id, str) else None

    voice_id = getattr(result, "voice_id", None)
    return voice_id if isinstance(voice_id, str) else None


async def _call_first_working_clone_method(method: object, attempts: list[dict]) -> object:
    """Try multiple call signatures for SDK compatibility and return first success."""
    if not callable(method):
        raise TypeError("Clone method is not callable")

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return await method(**kwargs)
        except TypeError as e:
            last_error = e
            continue

    if last_error:
        raise last_error
    raise RuntimeError("No clone method call attempts were provided")


def get_cloned_voice_id() -> str | None:
    """Return the active cloned voice ID, or None if not set."""
    return _cloned_voice_id


async def get_user_cloned_voice_id(user_id: str | None) -> str | None:
    """Return a user's cloned voice ID from profile storage, with global fallback."""
    if not user_id:
        return _cloned_voice_id

    from .user_profile import get_profile_service

    profile_service = get_profile_service()
    profile = await profile_service.get_profile(user_id)
    voice_id = profile.get("cloned_voice_id")
    if isinstance(voice_id, str) and voice_id.strip():
        return voice_id

    return _cloned_voice_id


async def get_clone_status(user_id: str | None = None) -> dict:
    """Return clone status, preferring user profile data when user_id is provided."""
    if user_id:
        from .user_profile import get_profile_service

        profile_service = get_profile_service()
        profile = await profile_service.get_profile(user_id)
        user_voice_id = profile.get("cloned_voice_id")
        user_voice_name = profile.get("cloned_voice_name")
        return {
            "is_cloned": bool(user_voice_id),
            "voice_id": user_voice_id,
            "voice_name": user_voice_name,
        }

    return {
        "is_cloned": _cloned_voice_id is not None,
        "voice_id": _cloned_voice_id,
        "voice_name": _cloned_voice_name,
    }


async def clear_cloned_voice(user_id: str | None = None) -> dict:
    """Reset cloned voice for a specific user, or clear global fallback voice."""
    if user_id:
        from .user_profile import get_profile_service

        profile_service = get_profile_service()
        profile = await profile_service.get_profile(user_id)
        old_id = profile.get("cloned_voice_id")
        await profile_service.save_profile(user_id, {
            "cloned_voice_id": None,
            "cloned_voice_name": "",
        })
        logger.info("voice_clone_cleared_for_user", user_id=user_id, old_voice_id=old_id)
        return {"status": "cleared", "old_voice_id": old_id}

    global _cloned_voice_id, _cloned_voice_name
    old_id = _cloned_voice_id
    _cloned_voice_id = None
    _cloned_voice_name = None
    logger.info("voice_clone_cleared", old_voice_id=old_id)
    return {"status": "cleared", "old_voice_id": old_id}


async def clone_voice(
    audio_bytes: bytes,
    filename: str,
    voice_name: str = "My Voice",
    user_id: str | None = None,
) -> dict:
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

        description = f"Cloned voice from {filename}"

        suffix = Path(filename).suffix or ".mp3"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            ivc_client = getattr(client.voices, "ivc", None)
            if ivc_client is not None:
                ivc_method = getattr(ivc_client, "create", None) or getattr(ivc_client, "add", None)
                if callable(ivc_method):
                    with open(tmp_path, "rb") as f:
                        result = await _call_first_working_clone_method(
                            ivc_method,
                            attempts=[
                                {"name": voice_name, "files": [f], "description": description},
                                {"name": voice_name, "files": [tmp_path], "description": description},
                                {"name": voice_name, "files": [audio_bytes], "description": description},
                                {"voice_name": voice_name, "files": [f], "description": description},
                                {"voice_name": voice_name, "files": [tmp_path], "description": description},
                            ],
                        )
                else:
                    result = None
            else:
                result = None

            if result is None:
                add_method = getattr(client.voices, "add", None)
                if callable(add_method):
                    with open(tmp_path, "rb") as f:
                        result = await _call_first_working_clone_method(
                            add_method,
                            attempts=[
                                {"name": voice_name, "files": [audio_bytes], "description": description},
                                {"name": voice_name, "files": [f], "description": description},
                                {"name": voice_name, "files": [tmp_path], "description": description},
                            ],
                        )

            if result is None:
                clone_method = getattr(client, "clone", None)
                if callable(clone_method):
                    result = await _call_first_working_clone_method(
                        clone_method,
                        attempts=[
                            {"name": voice_name, "files": [tmp_path], "description": description, "labels": "{}"},
                            {"name": voice_name, "files": [tmp_path], "description": description},
                            {"voice_name": voice_name, "files": [tmp_path], "description": description},
                        ],
                    )

            if result is None:
                available_methods = [m for m in dir(client.voices) if not m.startswith("_")]
                ivc_methods: list[str] = []
                if ivc_client is not None:
                    ivc_methods = [m for m in dir(ivc_client) if not m.startswith("_")]
                raise RuntimeError(
                    "Unsupported ElevenLabs SDK: no compatible voice clone method found "
                    f"(available voices methods: {available_methods}; ivc methods: {ivc_methods})"
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        _cloned_voice_id = _extract_voice_id(result)
        if not _cloned_voice_id:
            raise RuntimeError("Voice cloning succeeded but no voice_id was returned by ElevenLabs")
        _cloned_voice_name = voice_name

        logger.info(
            "voice_clone_success",
            voice_id=_cloned_voice_id,
            voice_name=voice_name,
        )

        result_payload = {
            "status": "success",
            "voice_id": _cloned_voice_id,
            "voice_name": voice_name,
        }

        if user_id:
            from .user_profile import get_profile_service

            profile_service = get_profile_service()
            await profile_service.save_profile(user_id, {
                "cloned_voice_id": _cloned_voice_id,
                "cloned_voice_name": voice_name,
            })
            result_payload["user_id"] = user_id

        return result_payload

    except Exception as e:
        logger.error(
            "voice_clone_failed",
            error=str(e),
            error_type=type(e).__name__,
            filename=filename,
        )
        raise
