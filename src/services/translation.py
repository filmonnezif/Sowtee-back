"""
Translation Service
Uses lingo.dev SDK for fast EN→AR translation before Arabic TTS.
"""

import os
import structlog

logger = structlog.get_logger(__name__)


async def translate_text(
    text: str,
    source_locale: str = "en",
    target_locale: str = "ar",
) -> str:
    """
    Translate text using lingo.dev's quick_translate (fast mode).

    Args:
        text: Text to translate.
        source_locale: Source language code.
        target_locale: Target language code (supports 'en', 'ar', 'ur').

    Returns:
        Translated text string.
    """
    from ..config import get_settings

    settings = get_settings()
    api_key = settings.lingodotdev_api_key or os.getenv("LINGODOTDEV_API_KEY", "")

    if not api_key:
        logger.error("LINGODOTDEV_API_KEY not set, returning original text")
        return text

    try:
        from lingodotdev.engine import LingoDotDevEngine

        result = await LingoDotDevEngine.quick_translate(
            text,
            api_key=api_key,
            source_locale=source_locale,
            target_locale=target_locale,
        )

        logger.info(
            "Translation complete",
            source=source_locale,
            target=target_locale,
            original=text[:100],
            translated=str(result)[:100],
        )

        return str(result)

    except Exception as e:
        logger.error("Translation failed, returning original text", error=str(e))
        return text
