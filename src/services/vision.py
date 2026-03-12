"""
SOWTEE Vision Service
Handles visual perception using Groq for multimodal analysis.
"""

import base64
import time
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from groq import AsyncGroq
from PIL import Image
import io

from ..config import get_settings
from ..models import DetectedObject, VisualContext
from .model_manager import Colors, log_model_input, log_model_output, log_model_error

logger = structlog.get_logger(__name__)


class VisionService:
    """
    Vision service using Groq for real-time scene understanding.
    
    Responsibilities:
    - Decode and process camera frames
    - Analyze visual context using multimodal LLM
    - Extract objects, scene description, and activity inference
    - Rate limiting to avoid quota exhaustion
    """
    
    # Groq vision model
    VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self._last_api_call: float = 0.0
        self._cached_context: VisualContext | None = None
        self._client: AsyncGroq | None = None
        self._configure_groq()
        
    def _configure_groq(self) -> None:
        """Configure the Groq API client."""
        if not self.settings.groq_api_key:
            logger.warning("Groq API key not configured - using mock responses")
            return
            
        self._client = AsyncGroq(api_key=self.settings.groq_api_key)
        logger.info("Groq Vision client configured", model=self.VISION_MODEL)
    
    def _decode_image(self, image_base64: str) -> Image.Image:
        """Decode base64 image to PIL Image."""
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        
        image_bytes = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_bytes))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
    )
    async def analyze_scene(self, image_base64: str) -> VisualContext:
        """
        Analyze a camera frame to extract visual context.
        
        This is the PERCEIVE phase of the agentic loop.
        Includes rate limiting to avoid quota exhaustion.
        
        Args:
            image_base64: Base64 encoded camera frame
            
        Returns:
            VisualContext with scene understanding
        """
        # Rate limiting: return cached result if called too frequently
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call
        
        if time_since_last_call < self.settings.min_frame_interval_seconds:
            if self._cached_context is not None:
                logger.debug(
                    "Rate limited, returning cached context",
                    wait_time=self.settings.min_frame_interval_seconds - time_since_last_call
                )
                return self._cached_context
        
        logger.debug("Analyzing visual scene")
        
        if self._client is None:
            return self._mock_visual_context()
        
        try:
            self._last_api_call = current_time
            start_time = time.time()
            
            # Remove data URL prefix if present
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            
            # Structured prompt for consistent output
            prompt = """Analyze this image for an AAC (Augmentative and Alternative Communication) system.
            
The user has motor/speech impairment and needs help communicating about their surroundings.

Provide a JSON response with:
1. "scene_description": A concise description of the scene (1-2 sentences)
2. "objects": Array of objects with {"label": string, "confidence": 0-1, "attributes": [strings]}
3. "environmental_context": The setting/location (e.g., "kitchen", "living room", "outdoors")
4. "activity_inference": What activity the user might be engaged in or wanting to do

Focus on actionable objects the user might want to interact with or communicate about.

Respond ONLY with valid JSON, no markdown."""

            # Log input
            log_model_input("vision", self.VISION_MODEL, prompt)
            
            # Build multimodal message for Groq
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]

            response = await self._client.chat.completions.create(
                model=self.VISION_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            
            response_text = response.choices[0].message.content or ""
            
            # Log output
            duration_ms = (time.time() - start_time) * 1000
            log_model_output("vision", self.VISION_MODEL, response_text, duration_ms)
            
            result = self._parse_vision_response(response_text)
            self._cached_context = result  # Cache for rate limiting
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            log_model_error("vision", self.VISION_MODEL, str(e))
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                logger.warning("Rate limited by Groq API, will retry", error=str(e))
                raise
            logger.error("Vision analysis failed", error=str(e))
            # Return cached context if available during errors
            if self._cached_context is not None:
                logger.info("Returning cached context due to error")
                return self._cached_context
            raise
    
    def _parse_vision_response(self, response_text: str) -> VisualContext:
        """Parse Gemini's JSON response into VisualContext."""
        import json
        
        try:
            # Clean potential markdown code blocks
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            
            data = json.loads(text)
            
            detected_objects = [
                DetectedObject(
                    label=obj.get("label", "unknown"),
                    confidence=float(obj.get("confidence", 0.5)),
                    attributes=obj.get("attributes", [])
                )
                for obj in data.get("objects", [])
            ]
            
            return VisualContext(
                scene_description=data.get("scene_description", "Unable to analyze scene"),
                detected_objects=detected_objects,
                environmental_context=data.get("environmental_context", ""),
                activity_inference=data.get("activity_inference", ""),
            )
            
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse vision response as JSON", error=str(e))
            # Fallback: use raw text as description
            return VisualContext(
                scene_description=response_text[:500],
                detected_objects=[],
                environmental_context="",
                activity_inference="",
            )
    
    def _mock_visual_context(self) -> VisualContext:
        """Return mock visual context for development without API key."""
        return VisualContext(
            scene_description="A living room with a person seated on a couch. A glass of water and remote control are on the coffee table.",
            detected_objects=[
                DetectedObject(label="couch", confidence=0.95, attributes=["furniture", "seating"]),
                DetectedObject(label="water glass", confidence=0.88, attributes=["drink", "beverage"]),
                DetectedObject(label="remote control", confidence=0.82, attributes=["device", "TV"]),
                DetectedObject(label="coffee table", confidence=0.90, attributes=["furniture"]),
            ],
            environmental_context="living room",
            activity_inference="relaxing, watching TV",
        )


# Singleton instance
_vision_service: VisionService | None = None


def get_vision_service() -> VisionService:
    """Get the vision service singleton."""
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service
