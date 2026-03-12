"""
SOWTEE Pydantic Schemas
Core data models for the agentic communication bridge.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class InteractionMode(str, Enum):
    """User interaction mode."""
    TOUCH = "touch"
    EYE_GAZE = "eye_gaze"
    SWITCH = "switch"


class AgentPhase(str, Enum):
    """Agentic loop phases."""
    PERCEIVE = "perceive"
    REASON = "reason"
    ACT = "act"
    LEARN = "learn"


# ============== Vision Models ==============

class DetectedObject(BaseModel):
    """An object detected in the visual scene."""
    label: str = Field(..., description="Object label/name")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: dict[str, float] | None = Field(
        default=None, 
        description="Normalized bounding box {x, y, width, height}"
    )
    attributes: list[str] = Field(default_factory=list)


class VisualContext(BaseModel):
    """Analyzed visual context from camera feed."""
    scene_description: str = Field(..., description="Natural language scene description")
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    environmental_context: str = Field(default="", description="Location/setting inference")
    activity_inference: str = Field(default="", description="Inferred user activity")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============== Memory Models ==============

class MemoryRecord(BaseModel):
    """A single memory record in the vector database."""
    id: UUID = Field(default_factory=uuid4)
    user_id: str
    scene_embedding_id: str | None = None
    visual_context_summary: str
    selected_phrase: str
    objects_present: list[str] = Field(default_factory=list)
    environmental_context: str = ""
    selection_count: int = 1
    last_used: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============== Prediction Models ==============

class PhraseCandidate(BaseModel):
    """A predicted phrase candidate for the user."""
    phrase: str
    phrase_arabic: str | None = Field(default=None, description="Arabic translation")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Why this phrase was predicted")
    source: str = Field(default="vision", description="Prediction source: vision, memory, hybrid")
    emotion: str | None = Field(default=None, description="Suggested vocal emotion, e.g. [cheerful]")
    related_objects: list[str] = Field(default_factory=list)


class ContextAnalysis(BaseModel):
    """Complete context analysis from the agentic orchestrator."""
    visual_context: VisualContext
    retrieved_memories: list[MemoryRecord] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    processing_time_ms: float = 0.0


class PredictionMode(str, Enum):
    """Mode for prediction optimization."""
    FULL = "full"  # Run full pipeline: vision + memory + intent
    VISION_ONLY = "vision_only"  # Only update visual context, skip intent
    ABBREVIATION = "abbreviation"  # User is typing abbreviations, skip intent


class PredictionRequest(BaseModel):
    """Request for phrase prediction."""
    user_id: str = Field(..., description="Unique user identifier")
    image_base64: str = Field(..., description="Base64 encoded camera frame")
    interaction_mode: InteractionMode = InteractionMode.TOUCH
    session_id: str | None = None
    additional_context: str = Field(default="", description="Optional user-provided context")
    mode: PredictionMode = Field(
        default=PredictionMode.FULL,
        description="Optimization mode: 'full' runs all phases, 'vision_only' skips intent, 'abbreviation' skips intent for typing mode"
    )


class PredictionResponse(BaseModel):
    """Response with predicted phrases."""
    session_id: str
    phrases: list[PhraseCandidate]
    context_analysis: ContextAnalysis
    agent_phase: AgentPhase = AgentPhase.ACT
    processing_time_ms: float


# ============== Feedback Models ==============

class UserFeedback(BaseModel):
    """User feedback when selecting a phrase."""
    session_id: str
    user_id: str
    selected_phrase: str
    was_from_predictions: bool = True
    custom_phrase: str | None = Field(
        default=None, 
        description="If user typed a custom phrase instead"
    )
    visual_context_summary: str = ""
    objects_present: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============== Agent State ==============

class AgentState(BaseModel):
    """Current state of the agentic orchestrator."""
    session_id: str
    user_id: str
    current_phase: AgentPhase = AgentPhase.PERCEIVE
    cycle_count: int = 0
    visual_context: VisualContext | None = None
    retrieved_memories: list[MemoryRecord] = Field(default_factory=list)
    candidate_phrases: list[PhraseCandidate] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# ============== Word Suggestion Models ==============

class WordCategory(str, Enum):
    """Category of a word in sentence building."""
    STARTER = "starter"
    SUBJECT = "subject"
    VERB = "verb"
    OBJECT = "object"
    MODIFIER = "modifier"
    ENDING = "ending"


class WordSuggestion(BaseModel):
    """A single word suggestion for sentence building."""
    word: str
    word_arabic: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    category: WordCategory
    related_to_scene: bool = False


class WordSuggestionRequest(BaseModel):
    """Request for word suggestions during sentence building."""
    user_id: str = Field(..., description="Unique user identifier")
    image_base64: str | None = Field(default=None, description="Optional base64 encoded camera frame")
    current_sentence: list[str] = Field(default_factory=list, description="Words already in the sentence")
    scene_context: str | None = Field(default=None, description="Previously analyzed scene context")
    conversation_context: list[str] = Field(default_factory=list, description="Recent transcribed speech from surroundings")
    interaction_mode: InteractionMode = InteractionMode.TOUCH


class WordSuggestionResponse(BaseModel):
    """Response with word suggestions for sentence building."""
    session_id: str
    words: list[WordSuggestion]
    scene_description: str | None = None
    processing_time_ms: float = 0.0

