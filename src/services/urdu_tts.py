"""
Urdu TTS Service
Uses upliftai API for high-quality Urdu text-to-speech.
"""

import os
import structlog
import httpx
from typing import AsyncGenerator

logger = structlog.get_logger(__name__)

# Uplift AI API configuration
UPLIFTAI_API_URL = "https://api.upliftai.org/v1/synthesis/text-to-speech"
DEFAULT_VOICE_ID = "v_8eelc901"  # Default Urdu voice from Uplift AI docs


async def generate_urdu_speech(text: str) -> AsyncGenerator[bytes, None]:
    """
    Generate Urdu speech using Uplift AI API.
    
    Args:
        text: Urdu text to convert to speech.
        
    Yields:
        Audio bytes chunks.
        
    Raises:
        Exception: If TTS generation fails.
    """
    from ..config import get_settings
    from .error_logger import get_error_logger
    
    error_logger = get_error_logger()
    
    logger.info("Starting Urdu TTS generation", text_length=len(text), text_preview=text[:100])
    
    settings = get_settings()
    api_key = settings.upliftai_api_key or os.getenv("UPLIFTAI_API_KEY", "")
    
    if not api_key:
        logger.error("UPLIFTAI_API_KEY not set")
        error_logger.log_error(
            error_type="urdu_tts_config",
            message="UPLIFTAI_API_KEY not configured",
            details={"text_length": len(text)}
        )
        raise ValueError("UPLIFTAI_API_KEY not configured")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "voiceId": DEFAULT_VOICE_ID,
        "text": text,
        "outputFormat": "MP3_22050_128"
    }
    
    logger.info("Making Uplift AI TTS request", voice_id=DEFAULT_VOICE_ID, payload=payload)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                UPLIFTAI_API_URL,
                headers=headers,
                json=payload
            )
            
            logger.info(
                "Uplift AI TTS response received",
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
            if response.status_code != 200:
                error_details = {
                    "status_code": response.status_code,
                    "response_text": response.text[:1000],
                    "headers": dict(response.headers),
                    "payload": payload
                }
                
                logger.error(
                    "Uplift AI TTS request failed",
                    **error_details
                )
                
                error_logger.log_error(
                    error_type="urdu_tts_api_error",
                    message=f"Uplift AI TTS failed with status {response.status_code}",
                    details=error_details
                )
                
                raise Exception(f"Uplift AI TTS failed: {response.status_code} - {response.text[:200]}")
            
            # Get audio duration from headers
            audio_duration = response.headers.get('x-uplift-ai-audio-duration')
            if audio_duration:
                logger.info(f"Urdu TTS audio duration: {audio_duration}ms")
            
            # Check if we got any content
            audio_data = response.content
            if not audio_data:
                logger.error("Uplift AI TTS returned empty response")
                error_logger.log_error(
                    error_type="urdu_tts_empty_response",
                    message="Uplift AI TTS returned empty audio data",
                    details={"status_code": response.status_code}
                )
                raise Exception("Uplift AI TTS returned empty audio data")
            
            logger.info(
                "Urdu TTS generation successful",
                text_length=len(text),
                audio_size=len(audio_data),
                audio_duration=audio_duration
            )
            
            # Yield the audio data as chunks
            chunk_size = 8192  # 8KB chunks
            chunks_sent = 0
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                yield chunk
                chunks_sent += 1
                
            logger.info(
                "Urdu TTS audio streaming complete",
                total_chunks=chunks_sent,
                total_bytes=len(audio_data)
            )
                
    except httpx.TimeoutException as e:
        logger.error("Uplift AI TTS request timed out", timeout=30.0, error=str(e))
        error_logger.log_error(
            error_type="urdu_tts_timeout",
            message="Uplift AI TTS request timed out",
            details={"timeout": 30.0, "error": str(e)}
        )
        raise Exception("Uplift AI TTS request timed out") from e
    except httpx.RequestError as e:
        logger.error("Uplift AI TTS network error", error=str(e), error_type=type(e).__name__)
        error_logger.log_error(
            error_type="urdu_tts_network_error",
            message=f"Uplift AI TTS network error: {type(e).__name__}",
            details={"error": str(e)}
        )
        raise Exception(f"Uplift AI TTS network error: {str(e)}") from e
    except Exception as e:
        logger.error("Uplift AI TTS generation failed", error=str(e), error_type=type(e).__name__)
        error_logger.log_error(
            error_type="urdu_tts_general_error",
            message=f"Urdu TTS generation failed: {type(e).__name__}",
            details={"error": str(e), "text_length": len(text)}
        )
        raise
