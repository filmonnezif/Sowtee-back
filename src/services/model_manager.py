"""
SOWTEE Model Manager
Centralized AI model management with Groq API support.
Uses best-in-class models for each task with colored logging.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import base64

import structlog
from groq import AsyncGroq

from .error_logger import get_error_logger

logger = structlog.get_logger(__name__)

# ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def log_model_input(task_type: str, model_id: str, prompt: str, system_prompt: str | None = None):
    """Log model input with cyan color."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}📥 MODEL INPUT [{task_type.upper()}] - {model_id}{Colors.RESET}")
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.RESET}")
    if system_prompt:
        print(f"{Colors.YELLOW}📋 System Prompt:{Colors.RESET}")
        print(f"{Colors.YELLOW}{system_prompt[:500]}{'...' if len(system_prompt) > 500 else ''}{Colors.RESET}")
        print()
    print(f"{Colors.CYAN}💬 User Prompt:{Colors.RESET}")
    print(f"{Colors.CYAN}{prompt[:1000]}{'...' if len(prompt) > 1000 else ''}{Colors.RESET}")
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.RESET}\n")


def log_model_output(task_type: str, model_id: str, output: str, duration_ms: float = 0):
    """Log model output with green color."""
    print(f"\n{Colors.GREEN}{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"{Colors.GREEN}{Colors.BOLD}📤 MODEL OUTPUT [{task_type.upper()}] - {model_id} ({duration_ms:.0f}ms){Colors.RESET}")
    print(f"{Colors.GREEN}═══════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"{Colors.GREEN}{output[:2000]}{'...' if len(output) > 2000 else ''}{Colors.RESET}")
    print(f"{Colors.GREEN}═══════════════════════════════════════════════════════════════{Colors.RESET}\n")


def log_model_error(task_type: str, model_id: str, error: str):
    """Log model error with red color."""
    print(f"\n{Colors.RED}{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"{Colors.RED}{Colors.BOLD}❌ MODEL ERROR [{task_type.upper()}] - {model_id}{Colors.RESET}")
    print(f"{Colors.RED}═══════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"{Colors.RED}{error}{Colors.RESET}")
    print(f"{Colors.RED}═══════════════════════════════════════════════════════════════{Colors.RESET}\n")


@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    model_id: str
    description: str
    max_tokens: int = 8192
    temperature: float = 0.7
    supports_vision: bool = False


@dataclass
class ModelUsage:
    """Track usage statistics for a model."""
    requests: int = 0
    errors: int = 0
    rate_limit_hits: int = 0
    last_error: str | None = None
    in_cooldown: bool = False
    cooldown_until: datetime | None = None


class ModelManager:
    """
    Centralized manager for AI model interactions using Groq.
    
    Uses best models for each task:
    - Vision: Llama 4 Scout (multimodal, 17B params, 16 experts)
    - Abbreviation/Reasoning: Qwen3-32B (best for complex reasoning)
    - Suggestions: Llama 3.3 70B (fast, versatile)
    """
    
    # Best models for different tasks (Groq)
    DEFAULT_MODELS: dict[str, str] = {
        "vision": "meta-llama/llama-4-scout-17b-16e-instruct",  # Multimodal vision model
        "abbreviation": "qwen/qwen3-32b",  # Best for reasoning/expansion
        "suggestions": "llama-3.3-70b-versatile",  # Fast and versatile
        "predictions": "llama-3.3-70b-versatile",  # Predictive text suggestions
        "formatting": "llama-3.1-8b-instant",  # Text cleanup (fast, cheap)
    }
    
    # Fallback model when primary hits rate limits
    FALLBACK_MODEL = "llama-3.1-8b-instant"  # Fastest fallback
    
    # Cooldown period after rate limit (seconds)
    RATE_LIMIT_COOLDOWN = 60.0
    
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: AsyncGroq | None = None
        self._models: dict[str, ModelConfig] = {}
        self._usage: dict[str, ModelUsage] = {}
        self._error_logger = get_error_logger()
        
        # Initialize default model configs
        self._init_default_models()
    
    def _init_default_models(self) -> None:
        """Initialize default model configurations."""
        self._models = {
            # Vision model - Llama 4 Scout (multimodal)
            "meta-llama/llama-4-scout-17b-16e-instruct": ModelConfig(
                model_id="meta-llama/llama-4-scout-17b-16e-instruct",
                description="Llama 4 Scout - Multimodal vision model, 17B params, 16 experts",
                max_tokens=8192,
                temperature=0.7,
                supports_vision=True,
            ),
            # Alternative vision model - Llama 4 Maverick (128 experts, higher quality)
            "meta-llama/llama-4-maverick-17b-128e-instruct": ModelConfig(
                model_id="meta-llama/llama-4-maverick-17b-128e-instruct",
                description="Llama 4 Maverick - Multimodal, 17B params, 128 experts",
                max_tokens=8192,
                temperature=0.7,
                supports_vision=True,
            ),
            # Best reasoning model - Qwen3-32B
            "qwen/qwen3-32b": ModelConfig(
                model_id="qwen/qwen3-32b",
                description="Qwen3-32B - Excellent for reasoning and complex tasks",
                max_tokens=8192,
                temperature=0.7,
                supports_vision=False,
            ),
            # Versatile model - Llama 3.3 70B
            "llama-3.3-70b-versatile": ModelConfig(
                model_id="llama-3.3-70b-versatile",
                description="Llama 3.3 70B - High quality, versatile",
                max_tokens=8192,
                temperature=0.7,
                supports_vision=False,
            ),
            # Fast fallback - Llama 3.1 8B
            "llama-3.1-8b-instant": ModelConfig(
                model_id="llama-3.1-8b-instant",
                description="Llama 3.1 8B - Ultra-fast inference",
                max_tokens=8192,
                temperature=0.7,
                supports_vision=False,
            ),
            # TTS Models (PlayAI / Canopy Labs on Groq)
            "playai-3.0-mini": ModelConfig(  # Assuming this is the exposed ID, referencing PlayAI Dialog
                model_id="playai-3.0-mini", # Or playai-dialog, need to be careful. I will trust the search result or use a more generic id. 
                # Actually, search said "PlayAI's Dialog model". 
                # Let's use a safe bet or a map. 
                # I'll stick to the text generation models here and handle TTS models in a separate dict as previously planned or just inline.
                description="PlayAI Dialog - Fast TTS",
                max_tokens=0,
                temperature=0.7,
                supports_vision=False,
            ),
        }
        
        # TTS Model Map
        self.TTS_MODELS = {
            "en": "canopylabs/orpheus-v1-english",
            "ar": "canopylabs/orpheus-arabic-saudi",
        }
        
        # Default voices per model/language
        self.TTS_VOICES = {
            "en": "austin", # User requested Austin
            "ar": "fahad",  # Male Saudi voice
        }
    
    def _get_client(self) -> AsyncGroq:
        """Get or create the Groq client."""
        if self._client is None:
            import os
            from ..config import get_settings
            settings = get_settings()
            api_key = self._api_key or settings.groq_api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set. Please set groq_api_key in .env")
            self._client = AsyncGroq(api_key=api_key)
        return self._client
    
    async def generate_speech(self, text: str, language: str = "en", emotion: str | None = None, enriched: bool = False) -> Any:
        """
        Generate speech from text using Groq TTS.
        Returns a streaming response object.
        
        Args:
            text: Text to speak (may already contain [directions] if enriched=True).
            language: Language code ("en" or "ar").
            emotion: Optional single emotion override (ignored if enriched=True).
            enriched: If True, text already has inline vocal directions from the enricher.
        """
        client = self._get_client()
        
        # Determine language code
        lang_code = "ar" if language.startswith("ar") else "en"
        
        if enriched:
            # Text already contains inline Orpheus vocal directions
            final_text = text
        else:
            # Prepend emotion to text (fallback to [happy] for stable, normal speech)
            target_emotion = emotion if emotion else "[happy]"
            
            valid_emotion = target_emotion.strip()
            if not valid_emotion.startswith("["):
                valid_emotion = f"[{valid_emotion}]"
                
            final_text = f"{valid_emotion} {text}"
        
        # Select model and voice
        model = self.TTS_MODELS.get(lang_code)
        voice = self.TTS_VOICES.get(lang_code, "austin")
        
        log_model_input("tts", model, final_text)
        
        start_time = datetime.now()
        
        try:
            # Use streaming response for low latency
            async with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=final_text,
                response_format="wav",
            ) as response:
                async for chunk in response.iter_bytes():
                    yield chunk
                
        except Exception as e:
            msg = str(e)
            log_model_error("tts", model, msg)
            self._record_error(model, msg, "tts")
            raise
    
    def _get_model_for_task(self, task_type: str) -> str:
        """Get the appropriate model for a task type."""
        return self.DEFAULT_MODELS.get(task_type, self.DEFAULT_MODELS["suggestions"])
    
    def _is_in_cooldown(self, model_id: str) -> bool:
        """Check if a model is in cooldown period."""
        usage = self._usage.get(model_id)
        if not usage or not usage.in_cooldown:
            return False
        
        if usage.cooldown_until and datetime.now() >= usage.cooldown_until:
            # Cooldown expired
            usage.in_cooldown = False
            usage.cooldown_until = None
            return False
        
        return True
    
    def _enter_cooldown(self, model_id: str) -> None:
        """Put a model into cooldown period."""
        if model_id not in self._usage:
            self._usage[model_id] = ModelUsage()
        
        usage = self._usage[model_id]
        usage.in_cooldown = True
        usage.cooldown_until = datetime.now() + timedelta(seconds=self.RATE_LIMIT_COOLDOWN)
        usage.rate_limit_hits += 1
        
        logger.warning(
            "Model rate limited, entering cooldown",
            model=model_id,
            cooldown_seconds=self.RATE_LIMIT_COOLDOWN,
        )
    
    async def generate(
        self,
        task_type: str,
        prompt: str,
        image_base64: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate text using the appropriate model for the task.
        
        Args:
            task_type: Type of task (vision, abbreviation, suggestions)
            prompt: The prompt to send
            image_base64: Optional base64 encoded image (for vision tasks)
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        primary_model = self._get_model_for_task(task_type)
        
        # Check if primary model is in cooldown
        if self._is_in_cooldown(primary_model):
            logger.info("Primary model in cooldown, using fallback", 
                       primary=primary_model, fallback=self.FALLBACK_MODEL)
            model_id = self.FALLBACK_MODEL
        else:
            model_id = primary_model
        
        # If fallback is also in cooldown, wait or raise
        if self._is_in_cooldown(model_id):
            raise RuntimeError("All models in cooldown, please try again later")
        
        # Check if we need vision and model supports it
        has_image = image_base64 is not None and task_type == "vision"
        model_config = self._models.get(model_id)
        
        if has_image and model_config and not model_config.supports_vision:
            # Switch to vision model if we have an image
            model_id = self.DEFAULT_MODELS["vision"]
        
        try:
            return await self._generate_with_model(
                model_id, prompt, system_prompt, task_type, image_base64 if has_image else None
            )
        except Exception as e:
            error_str = str(e)
            
            # Check for rate limit
            if "rate" in error_str.lower() or "429" in error_str:
                self._enter_cooldown(model_id)
                
                # Try fallback if we haven't already
                if model_id != self.FALLBACK_MODEL and not self._is_in_cooldown(self.FALLBACK_MODEL):
                    logger.info("Trying fallback model after rate limit", 
                               fallback=self.FALLBACK_MODEL)
                    return await self._generate_with_model(
                        self.FALLBACK_MODEL, prompt, system_prompt, task_type, None
                    )
            
            # Log the error
            self._record_error(model_id, error_str, task_type)
            raise
    
    async def _generate_with_model(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str | None,
        task_type: str,
        image_base64: str | None = None,
    ) -> str:
        """Generate text using a specific model."""
        import time
        start_time = time.time()
        
        client = self._get_client()
        
        # Track usage
        if model_id not in self._usage:
            self._usage[model_id] = ModelUsage()
        self._usage[model_id].requests += 1
        
        # Log input with color
        log_model_input(task_type, model_id, prompt, system_prompt)
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Handle vision content
        if image_base64:
            # Multimodal message with image
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        # Get model config
        config = self._models.get(model_id)
        temperature = config.temperature if config else 0.7
        max_tokens = min(config.max_tokens if config else 4096, 4096)  # Cap at 4096 for speed
        
        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            result = response.choices[0].message.content or ""
            
            # Calculate duration and log output with color
            duration_ms = (time.time() - start_time) * 1000
            log_model_output(task_type, model_id, result, duration_ms)
            
            logger.debug(
                "Model generation complete",
                model=model_id,
                task_type=task_type,
                response_length=len(result),
            )
            
            return result
            
        except Exception as e:
            self._usage[model_id].errors += 1
            log_model_error(task_type, model_id, str(e))
            raise
    
    def _record_error(self, model_id: str, error: str, task_type: str) -> None:
        """Record an error for a model."""
        if model_id not in self._usage:
            self._usage[model_id] = ModelUsage()
        
        usage = self._usage[model_id]
        usage.errors += 1
        usage.last_error = error
        
        logger.error(
            "Model error",
            model=model_id,
            task_type=task_type,
            error=error,
            total_errors=usage.errors,
        )
        
        # Use log_error with correct signature
        self._error_logger.log_error(
            source="model_manager",
            error_type="model_error",
            message=f"Model {model_id} failed: {error}",
            model=model_id,
            details={"task_type": task_type},
        )
    
    def get_usage_stats(self) -> dict[str, dict]:
        """Get usage statistics for all models."""
        return {
            model_id: {
                "requests": usage.requests,
                "errors": usage.errors,
                "rate_limit_hits": usage.rate_limit_hits,
                "last_error": usage.last_error,
                "in_cooldown": usage.in_cooldown,
            }
            for model_id, usage in self._usage.items()
        }
    
    def get_available_models(self) -> dict[str, dict]:
        """Get list of available models and their capabilities."""
        return {
            model_id: {
                "description": config.description,
                "max_tokens": config.max_tokens,
                "supports_vision": config.supports_vision,
            }
            for model_id, config in self._models.items()
        }
    
    def reset_cooldowns(self) -> None:
        """Reset all model cooldowns (for testing/admin)."""
        for usage in self._usage.values():
            usage.in_cooldown = False
            usage.cooldown_until = None
        logger.info("All model cooldowns reset")


# Singleton instance
_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get the global ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
