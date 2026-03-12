"""
SOWTEE Configuration Module
Centralized settings management with environment variable support.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    app_name: str = "SOWTEE"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://127.0.0.1:3000"])
    
    # Gemini API Configuration (legacy)
    gemini_api_key: str = Field(default="", description="Google Gemini API Key")
    gemini_vision_model: str = "gemini-2.0-flash"  # Multimodal model for scene analysis
    gemini_text_model: str = "gemini-2.0-flash-lite"  # Text-only for higher rate limits
    gemini_max_tokens: int = 1024
    gemini_temperature: float = 0.7
    
    # Groq API Configuration (primary for text generation)
    groq_api_key: str = Field(default="", description="Groq API Key")
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"
    
    # Lingo.dev Translation API (for EN→AR/UR translation)
    lingodotdev_api_key: str = Field(default="", description="Lingo.dev API Key for translation")
    
    # Uplift AI Urdu TTS API
    upliftai_api_key: str = Field(default="", description="Uplift AI API Key for Urdu TTS")
    
    # ElevenLabs TTS API
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API Key for TTS")
    elevenlabs_voice_id: str = Field(default="JBFqnCBsd6RMkjVDRZzb", description="ElevenLabs voice ID")
    elevenlabs_model_id: str = Field(default="eleven_flash_v2_5", description="ElevenLabs model ID")
    
    # Rate limiting (to avoid quota exhaustion)
    min_frame_interval_seconds: float = 2.0  # Minimum seconds between API calls
    
    # Vector Database (ChromaDB)
    chroma_persist_directory: str = "./data/chroma"
    chroma_collection_name: str = "sowtee_memory"
    
    # Agentic Orchestrator
    max_reasoning_cycles: int = 3
    prediction_confidence_threshold: float = 0.6
    top_k_phrases: int = 3
    memory_retrieval_limit: int = 10
    
    # User Session
    session_timeout_minutes: int = 60
    max_history_items: int = 100


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
