"""
SOWTEE FastAPI Application
Main application entry point with API routes.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Literal

import structlog
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import get_settings
from .models import (
    PredictionRequest, 
    PredictionResponse, 
    UserFeedback, 
    AgentState,
    WordSuggestionRequest,
    WordSuggestionResponse,
)
from .services.orchestrator import get_orchestrator
from .services.startup_checks import run_startup_key_checks

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info(
        "SOWTEE starting",
        version=settings.app_version,
        debug=settings.debug,
    )

    startup_keys_ok = await run_startup_key_checks(settings)
    if not startup_keys_ok:
        logger.error(
            "Startup key checks failed for required providers",
            strict_mode=settings.startup_key_check_strict,
        )
        if settings.startup_key_check_strict:
            raise RuntimeError("Startup key checks failed for required providers")
    
    # Initialize services
    _ = get_orchestrator()
    
    yield
    
    logger.info("SOWTEE shutting down")


app = FastAPI(
    title="SOWTEE API",
    description=(
        "صوتي (SOWTEE) - The Self-Learning, Context-Aware Agentic Communication Bridge. "
        "An AAC platform for patients with MS and ALS using multimodal AI."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Health & Status ==============

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, str]


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check system health and service status."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        services={
            "orchestrator": "active",
            "vision": "active" if settings.gemini_api_key else "mock",
            "memory": "active",
            "intent": "active",
        },
    )


# ============== Prediction API ==============

@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Get phrase predictions from camera frame",
)
async def predict_phrases(request: PredictionRequest) -> PredictionResponse:
    """
    Process a camera frame and return predicted phrases.
    
    This endpoint triggers the full agentic loop:
    1. **PERCEIVE**: Analyze the image using Gemini Vision
    2. **REASON**: Retrieve memories and synthesize intent
    3. **ACT**: Generate and return phrase predictions
    
    The response includes:
    - Top-k predicted phrases with confidence scores
    - Visual context analysis
    - Reasoning trace for transparency
    """
    orchestrator = get_orchestrator()
    return await orchestrator.process_frame(request)


# ============== Word Suggestion API ==============

@app.post(
    "/api/v1/suggest-words",
    response_model=WordSuggestionResponse,
    tags=["Prediction"],
    summary="Get word suggestions for sentence building",
)
async def suggest_words(request: WordSuggestionRequest) -> WordSuggestionResponse:
    """
    Get intelligent word suggestions for incremental sentence building.
    
    This endpoint uses AI to provide context-aware suggestions based on:
    - The current visual scene (if image provided)
    - The sentence being built
    - User's history and preferences
    - Natural language patterns
    
    Returns both individual word suggestions and complete phrase suggestions.
    """
    from .services.word_suggestions import get_word_suggestion_service
    
    service = get_word_suggestion_service()
    return await service.get_suggestions(request)


# ============== Predictive Text API ==============


class TextPredictionRequest(BaseModel):
    """Request for predictive text suggestions."""
    user_id: str
    partial_text: str = ""
    scene_description: str | None = None
    conversation_history: list[dict] | None = None
    num_suggestions: int = 5
    language: str = "en"


class PredictedTextItem(BaseModel):
    text: str
    confidence: float
    is_completion: bool = False


class TextPredictionResponse(BaseModel):
    """Response with predictive text suggestions."""
    suggestions: list[PredictedTextItem]
    ghost_text: str | None = None
    processing_time_ms: float = 0.0


@app.post(
    "/api/v1/predict-text",
    response_model=TextPredictionResponse,
    tags=["Prediction"],
    summary="Get predictive text suggestions",
)
async def predict_text(request: TextPredictionRequest) -> TextPredictionResponse:
    """
    Get continuous predictive text suggestions based on context.

    Provides preemptive suggestions (when partial_text is empty) and
    completion suggestions (when user has typed partial text).

    Context sources:
    - Visual scene description (from camera)
    - Conversation history (what others said, what user said)
    - User's vocabulary history (learned preferences)
    """
    from .services.predictive_suggestions import get_predictive_suggestion_service

    service = get_predictive_suggestion_service()
    result = await service.get_predictions(
        user_id=request.user_id,
        partial_text=request.partial_text,
        scene_description=request.scene_description,
        conversation_history=request.conversation_history,
        num_suggestions=request.num_suggestions,
        language=request.language,
    )

    return TextPredictionResponse(
        suggestions=[
            PredictedTextItem(
                text=s.text,
                confidence=s.confidence,
                is_completion=s.is_completion,
            )
            for s in result.suggestions
        ],
        ghost_text=result.ghost_text,
        processing_time_ms=result.processing_time_ms,
    )


class AcceptSuggestionRequest(BaseModel):
    """Request for storing an accepted suggestion."""
    user_id: str
    accepted_text: str
    scene_description: str | None = None
    conversation_context: str | None = None


@app.post(
    "/api/v1/accept-suggestion",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Learning"],
    summary="Store accepted text prediction for learning",
)
async def accept_suggestion(request: AcceptSuggestionRequest) -> dict[str, str]:
    """
    Store an accepted suggestion so the system learns the user's preferences.
    """
    from .services.predictive_suggestions import get_predictive_suggestion_service

    service = get_predictive_suggestion_service()
    await service.store_accepted_suggestion(
        user_id=request.user_id,
        accepted_text=request.accepted_text,
        scene_description=request.scene_description,
        conversation_context=request.conversation_context,
    )

    return {"status": "accepted", "message": "Suggestion stored for learning"}


class FormatTextRequest(BaseModel):
    """Request for AI text formatting."""
    text: str
    user_id: str | None = None


class FormatTextResponse(BaseModel):
    """Response with formatted text."""
    formatted_text: str
    was_modified: bool


@app.post(
    "/api/v1/format-text",
    response_model=FormatTextResponse,
    tags=["Prediction"],
    summary="Auto-format text (capitalization, punctuation, spacing)",
)
async def format_text(request: FormatTextRequest) -> FormatTextResponse:
    """
    AI-powered text cleanup: fix capitalization, punctuation, and spacing.
    Returns the original text if formatting fails.
    """
    from .services.predictive_suggestions import get_predictive_suggestion_service

    service = get_predictive_suggestion_service()
    formatted = await service.format_text(request.text)

    return FormatTextResponse(
        formatted_text=formatted,
        was_modified=formatted != request.text,
    )


class SurroundingTranscriptionResponse(BaseModel):
    """Response from surrounding voice transcription."""
    text: str
    language: str | None = None
    duration: float | None = None
    model: str | None = None
    has_speech: bool = False
    avg_no_speech_prob: float | None = None
    avg_logprob: float | None = None
    speech_confidence: float | None = None


@app.post(
    "/api/v1/transcribe/surrounding",
    response_model=SurroundingTranscriptionResponse,
    tags=["Skills"],
    summary="Transcribe surrounding speech using Groq Whisper",
)
async def transcribe_surrounding_speech(
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> SurroundingTranscriptionResponse:
    """Transcribe a surrounding speech segment audio file."""
    from .services.model_manager import get_model_manager

    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is required",
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio payload is empty",
        )

    if len(audio_bytes) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio file too large (max 15MB)",
        )

    manager = get_model_manager()
    try:
        result = await manager.transcribe_audio(
            audio_bytes,
            filename=audio.filename or "surrounding.webm",
            content_type=audio.content_type,
            language=language,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "could not process file" in message or "valid media file" in message:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid audio format for transcription",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Transcription provider request failed",
        ) from exc

    text = (result.get("text") or "").strip()
    segments = result.get("segments") or []

    no_speech_values: list[float] = []
    logprob_values: list[float] = []

    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue

            no_speech_prob = segment.get("no_speech_prob")
            if isinstance(no_speech_prob, (int, float)):
                no_speech_values.append(float(no_speech_prob))

            avg_logprob = segment.get("avg_logprob")
            if isinstance(avg_logprob, (int, float)):
                logprob_values.append(float(avg_logprob))

    avg_no_speech_prob = (
        sum(no_speech_values) / len(no_speech_values)
        if no_speech_values
        else None
    )
    avg_logprob = (
        sum(logprob_values) / len(logprob_values)
        if logprob_values
        else None
    )

    speech_confidence: float | None = None
    if avg_no_speech_prob is not None:
        speech_confidence = max(0.0, min(1.0, 1.0 - avg_no_speech_prob))
        if avg_logprob is not None:
            logprob_signal = max(0.0, min(1.0, (avg_logprob + 1.5) / 2.5))
            speech_confidence = max(0.0, min(1.0, (speech_confidence * 0.7) + (logprob_signal * 0.3)))

    logger.info(
        "whisper_transcription_result",
        raw_text_len=len((result.get("text") or "").strip()),
        accepted_text_len=len(text),
        filtered_out=False,
        filter_reason="frontend_vad_guarded",
        model=result.get("model"),
        language=result.get("language"),
        duration=result.get("duration"),
        avg_no_speech_prob=avg_no_speech_prob,
        avg_logprob=avg_logprob,
        speech_confidence=speech_confidence,
    )

    return SurroundingTranscriptionResponse(
        text=text,
        language=result.get("language"),
        duration=result.get("duration"),
        model=result.get("model"),
        has_speech=bool(text),
        avg_no_speech_prob=avg_no_speech_prob,
        avg_logprob=avg_logprob,
        speech_confidence=speech_confidence,
    )


@app.post(
    "/api/v1/feedback",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Learning"],
    summary="Submit phrase selection feedback",
)
async def submit_feedback(feedback: UserFeedback) -> dict[str, str]:
    """
    Submit user feedback when a phrase is selected.
    
    This triggers the **LEARN** phase:
    - Stores the selection in the vector database
    - Associates the phrase with the visual context
    - Updates frequency counts for personalization
    
    The system uses this data to improve future predictions.
    """
    orchestrator = get_orchestrator()
    await orchestrator.process_feedback(feedback)
    return {"status": "accepted", "message": "Feedback stored for learning"}


@app.get(
    "/api/v1/session/{session_id}",
    response_model=AgentState | None,
    tags=["Session"],
    summary="Get current session state",
)
async def get_session(session_id: str) -> AgentState:
    """Get the current state of an active session."""
    orchestrator = get_orchestrator()
    state = orchestrator.get_session_state(session_id)
    
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    
    return state


@app.delete(
    "/api/v1/session/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Session"],
    summary="Clear session state",
)
async def clear_session(session_id: str) -> None:
    """Clear a session's state."""
    orchestrator = get_orchestrator()
    orchestrator.clear_session(session_id)


# ============== User Profile API ==============

class UserProfileRequest(BaseModel):
    """User profile data for saving."""
    display_name: str = ""
    age: int | None = None
    condition: str = ""
    condition_stage: str = ""
    primary_language: str = "en"
    secondary_language: str = ""
    location: str = ""
    living_situation: str = ""
    interests: list[str] = []
    daily_routine: str = ""
    communication_style: str = "casual"
    common_needs: list[str] = []
    caregiver_name: str = ""
    notes: str = ""
    cloned_voice_id: str | None = None
    cloned_voice_name: str = ""


@app.get(
    "/api/v1/profile/{user_id}",
    tags=["User"],
    summary="Get user profile",
)
async def get_user_profile(user_id: str) -> dict:
    """Get a user's profile data. Returns defaults if no profile exists."""
    from .services.user_profile import get_profile_service
    
    service = get_profile_service()
    return await service.get_profile(user_id)


@app.put(
    "/api/v1/profile/{user_id}",
    tags=["User"],
    summary="Save or update user profile",
)
async def save_user_profile(user_id: str, profile: UserProfileRequest) -> dict:
    """
    Save or update a user's profile.
    
    The agent uses this profile to personalize all predictions,
    suggestions, and communication — learning to become YOUR voice.
    """
    from .services.user_profile import get_profile_service
    
    service = get_profile_service()
    saved = await service.save_profile(user_id, profile.model_dump(exclude_none=True))
    return saved


# ============== Learning Dashboard ==============


@app.get(
    "/api/v1/learning/{user_id}/metrics",
    tags=["Learning"],
    summary="Get learning metrics",
)
async def get_learning_metrics(user_id: str, simulate: bool = True) -> dict:
    """Get all learning metrics for the dashboard — shows self-improvement."""
    from .services.learning_tracker import get_learning_tracker

    tracker = get_learning_tracker()
    return await tracker.get_metrics(user_id, simulate=simulate)


@app.get(
    "/api/v1/learning/{user_id}/timeline",
    tags=["Learning"],
    summary="Get improvement timeline",
)
async def get_learning_timeline(user_id: str, simulate: bool = True) -> list[dict]:
    """Get improvement timeline data for charting accuracy over time."""
    from .services.learning_tracker import get_learning_tracker

    tracker = get_learning_tracker()
    return await tracker.get_improvement_timeline(user_id, simulate=simulate)


@app.get(
    "/api/v1/learning/{user_id}/strategies",
    tags=["Learning"],
    summary="Get strategy performance",
)
async def get_strategy_stats(user_id: str, simulate: bool = True) -> dict:
    """Get per-strategy success rates — shows which approaches work best."""
    from .services.learning_tracker import get_learning_tracker

    tracker = get_learning_tracker()
    return await tracker.get_strategy_stats(user_id, simulate=simulate)


@app.get(
    "/api/v1/learning/{user_id}/events",
    tags=["Learning"],
    summary="Get learning events",
)
async def get_learning_events(user_id: str, limit: int = 50, simulate: bool = True) -> list[dict]:
    """Get recent learning events — the agent's learning feed."""
    from .services.learning_tracker import get_learning_tracker

    tracker = get_learning_tracker()
    return await tracker.get_learning_events(user_id, limit=limit, simulate=simulate)


@app.get(
    "/api/v1/learning/{user_id}/summary",
    tags=["Learning"],
    summary="Get improvement summary",
)
async def get_learning_summary(user_id: str, simulate: bool = True) -> dict:
    """Get human-readable improvement summary for the dashboard hero."""
    from .services.learning_tracker import get_learning_tracker

    tracker = get_learning_tracker()
    return await tracker.get_improvement_summary(user_id, simulate=simulate)


# ============== User History ==============

class PhraseFrequency(BaseModel):
    phrase: str
    count: int


@app.get(
    "/api/v1/users/{user_id}/phrases",
    response_model=list[PhraseFrequency],
    tags=["User"],
    summary="Get user's most used phrases",
)
async def get_user_phrases(user_id: str, limit: int = 20) -> list[PhraseFrequency]:
    """Get a user's most frequently used phrases."""
    from .services.memory import get_memory_service
    
    memory = get_memory_service()
    frequencies = await memory.get_user_phrase_frequencies(user_id, limit)
    
    return [
        PhraseFrequency(phrase=phrase, count=count)
        for phrase, count in frequencies
    ]


# ============== Skills API ==============

class SkillInfoResponse(BaseModel):
    skill_id: str
    name: str
    description: str
    icon: str
    status: str
    version: str


class SkillActionRequest(BaseModel):
    """Request for a skill action."""
    user_id: str
    session_id: str | None = None
    action: str
    language: str = "en"
    scene_description: str | None = None
    scene_image: str | None = None
    conversation_history: list[dict] | None = None
    # Action-specific parameters
    card_index: int | None = None
    letter_index: int | None = None
    count: int = 5


class SkillActionResponse(BaseModel):
    """Response from a skill action."""
    action: str
    state: dict | None = None
    cards: list[dict] | None = None
    spread_letters: list[dict] | None = None
    selected_letter: str | None = None
    grouped_options: list[str] | None = None
    typed_text: str | None = None
    expansion: dict | None = None
    suggestions: list[dict] | None = None
    error: str | None = None


@app.get(
    "/api/v1/skills",
    response_model=list[SkillInfoResponse],
    tags=["Skills"],
    summary="List available skills",
)
async def list_skills() -> list[SkillInfoResponse]:
    """List all available agent skills."""
    from .skills import get_skill_registry
    
    registry = get_skill_registry()
    skills = registry.list_skills()
    
    return [
        SkillInfoResponse(
            skill_id=skill.skill_id,
            name=skill.name,
            description=skill.description,
            icon=skill.icon,
            status=skill.status.value,
            version=skill.version,
        )
        for skill in skills
    ]


@app.post(
    "/api/v1/skills/speaking/action",
    response_model=SkillActionResponse,
    tags=["Skills"],
    summary="Execute a Speaking skill action",
)
async def speaking_skill_action(request: SkillActionRequest) -> SkillActionResponse:
    """
    Execute an action on the Speaking skill.
    
    Available actions:
    - **get_cards**: Get the 5 letter cards
    - **select_card**: Select a card (provide card_index 0-4)
    - **select_letter**: Select a letter from spread (provide letter_index 0-4)
    - **expand**: Expand typed abbreviation to sentences
    - **get_suggestions**: Get AI sentence suggestions
    - **reset**: Reset the letter system
    - **backspace**: Remove last character
    - **add_space**: Add space after abbreviation
    - **go_back**: Go from letters back to cards
    - **get_state**: Get current state
    """
    from .skills import get_skill_registry, SkillContext
    
    registry = get_skill_registry()
    await registry.initialize()
    
    skill = registry.get_skill("speaking")
    if not skill:
        return SkillActionResponse(action=request.action, error="Speaking skill not found")
    
    # Build context
    context = SkillContext(
        user_id=request.user_id,
        session_id=request.session_id or "default",
        scene_description=request.scene_description,
        scene_image=request.scene_image,
        conversation_history=request.conversation_history or [],
    )
    
    # Execute action
    result = await skill.process(
        context,
        action=request.action,
        card_index=request.card_index,
        letter_index=request.letter_index,
        count=request.count,
        language=request.language,
    )
    
    return SkillActionResponse(**result)


class ExpansionRequest(BaseModel):
    """Request for abbreviation expansion."""
    abbreviation: str
    user_id: str | None = None  # For personalized history lookup
    scene_description: str | None = None
    conversation_context: str | None = None
    custom_context: str | None = None  # User-defined situation (e.g., "Giving a speech at hackathon")
    num_suggestions: int = 5


class ExpansionResponse(BaseModel):
    """Response with expanded abbreviations."""
    abbreviation: str
    expansions: list[str]
    confidences: list[float]
    primary: str
    alternatives: list[str]


class ExpansionFeedbackRequest(BaseModel):
    """Request for storing abbreviation expansion feedback."""
    user_id: str
    abbreviation: str
    selected_expansion: str
    scene_description: str | None = None
    conversation_context: str | None = None


@app.post(
    "/api/v1/skills/speaking/expand",
    response_model=ExpansionResponse,
    tags=["Skills"],
    summary="Expand an abbreviation to sentences",
)
async def expand_abbreviation(request: ExpansionRequest) -> ExpansionResponse:
    """
    Expand a letter abbreviation to possible sentences.
    
    Example: "i w t s" → ["I want to sleep", "I want to stay", ...]
    
    If user_id is provided, the system will use the user's history
    to prioritize expansions they've selected before in similar contexts.
    """
    from .skills.speaking.abbreviation_expander import get_abbreviation_expander
    
    expander = get_abbreviation_expander()
    
    # Update scene cache if provided
    if request.scene_description:
        expander.update_scene_context(request.scene_description)
    
    result = await expander.expand(
        abbreviation=request.abbreviation,
        scene_description=request.scene_description,
        conversation_context=request.conversation_context,
        custom_context=request.custom_context,  # User-defined situation
        num_suggestions=request.num_suggestions,
        user_id=request.user_id,  # For personalized history
    )
    
    return ExpansionResponse(
        abbreviation=result.abbreviation,
        expansions=result.expansions,
        confidences=result.confidences,
        primary=result.primary,
        alternatives=result.alternatives,
    )


@app.post(
    "/api/v1/skills/speaking/expand/feedback",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Skills", "Learning"],
    summary="Submit abbreviation expansion feedback",
)
async def expansion_feedback(request: ExpansionFeedbackRequest) -> dict[str, str]:
    """
    Submit feedback when user selects an abbreviation expansion.
    
    This enables the system to learn the user's preferences and
    prioritize their commonly used expansions in future suggestions.
    Similar to how intent predictions learn from phrase selections.
    """
    from .skills.speaking.abbreviation_expander import get_abbreviation_expander
    
    expander = get_abbreviation_expander()
    
    await expander.store_selection(
        user_id=request.user_id,
        abbreviation=request.abbreviation,
        selected_expansion=request.selected_expansion,
        scene_description=request.scene_description,
        conversation_context=request.conversation_context,
    )
    
    return {"status": "accepted", "message": "Expansion feedback stored for learning"}


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    user_id: str | None = None
    voice_option: Literal["cloned", "male", "female"] = "male"
    emotion: str | None = None
    scene_description: str | None = None
    conversation_context: str | None = None
    custom_context: str | None = None
    enrich_directions: bool = True


@app.post(
    "/api/v1/tts/stream",
    tags=["Skills"],
    summary="Stream TTS audio",
    description="Stream audio from text using ElevenLabs TTS with emotion-aware voice settings."
)
async def tts_stream(request: TTSRequest):
    """Stream audio for the given text, with automatic emotion-aware voice settings and Arabic/Urdu translation."""
    from fastapi.responses import StreamingResponse
    from .services.tts_direction_enricher import get_voice_settings, DEFAULT_VOICE_SETTINGS
    from .services.elevenlabs_tts import generate_elevenlabs_speech
    from .services.urdu_tts import generate_urdu_speech
    
    is_arabic = request.language.startswith("ar")
    is_urdu = request.language.startswith("ur")
    
    if is_urdu:
        from .services.translation import translate_text

        # Urdu flow: translate EN→UR via lingo.dev, then use Uplift AI Urdu TTS
        translated_text = await translate_text(
            text=request.text,
            source_locale="en",
            target_locale="ur",
        )
        return StreamingResponse(
            generate_urdu_speech(translated_text),
            media_type="audio/mpeg",  # Uplift AI returns MP3
        )
    
    # English & Arabic: use ElevenLabs TTS with emotion-aware voice settings
    speak_text = request.text
    
    if is_arabic:
        from .services.translation import translate_text

        # Translate EN→AR first
        speak_text = await translate_text(
            text=request.text,
            source_locale="en",
            target_locale="ar",
        )
    
    # Get emotion-aware voice settings from the direction enricher
    if request.enrich_directions:
        voice_settings = await get_voice_settings(
            text=request.text,  # Analyze emotion from original English text
            scene_description=request.scene_description,
            conversation_context=request.conversation_context,
            custom_context=request.custom_context,
        )
    else:
        voice_settings = DEFAULT_VOICE_SETTINGS.copy()
    
    return StreamingResponse(
        generate_elevenlabs_speech(
            text=speak_text,
            language=request.language,
            user_id=request.user_id,
            voice_option=request.voice_option,
            stability=voice_settings["stability"],
            similarity_boost=voice_settings["similarity_boost"],
            style=voice_settings["style"],
            speed=voice_settings.get("speed", 1.0),
        ),
        media_type="audio/mpeg",  # ElevenLabs returns MP3
    )


# ──────────────── Voice Cloning ────────────────

@app.post(
    "/api/v1/voice/clone",
    tags=["Skills"],
    summary="Clone a voice from an audio sample",
    description="Upload a short audio file (10-30s) to create a cloned voice via ElevenLabs IVC.",
)
async def clone_voice_endpoint(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    voice_name: str = Form("My Voice"),
):
    """Clone a voice from an uploaded audio sample."""
    from .services.voice_clone import clone_voice

    # Validate file type
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Read and validate size (max 5MB)
    audio_bytes = await file.read()
    if len(audio_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")
    if len(audio_bytes) < 1024:
        raise HTTPException(status_code=400, detail="File too small — provide at least 10 seconds of audio")

    try:
        result = await clone_voice(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.mp3",
            voice_name=voice_name,
            user_id=user_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")


@app.get(
    "/api/v1/voice/clone",
    tags=["Skills"],
    summary="Get voice clone status",
)
async def get_voice_clone_status(user_id: str = Query(...)):
    """Check if a cloned voice is active."""
    from .services.voice_clone import get_clone_status
    return await get_clone_status(user_id=user_id)


@app.delete(
    "/api/v1/voice/clone",
    tags=["Skills"],
    summary="Remove cloned voice",
)
async def remove_cloned_voice(user_id: str = Query(...)):
    """Remove the cloned voice and revert to the default voice."""
    from .services.voice_clone import clear_cloned_voice
    return await clear_cloned_voice(user_id=user_id)


class ModelStatusResponse(BaseModel):
    """Response with model usage statistics."""
    models: dict[str, dict]


@app.get(
    "/api/v1/models/status",
    response_model=ModelStatusResponse,
    tags=["System"],
    summary="Get AI model status and usage",
)
async def get_model_status() -> ModelStatusResponse:
    """Get usage statistics and status for all AI models."""
    from .services.model_manager import get_model_manager
    
    manager = get_model_manager()
    return ModelStatusResponse(models=manager.get_usage_stats())


class RecentErrorsResponse(BaseModel):
    """Response with recent errors."""
    errors: list[dict]
    counts: dict[str, int]


@app.get(
    "/api/v1/errors/recent",
    response_model=RecentErrorsResponse,
    tags=["System"],
    summary="Get recent errors",
)
async def get_recent_errors(limit: int = 20) -> RecentErrorsResponse:
    """Get recent errors for debugging."""
    from .services.error_logger import get_error_logger
    
    error_logger = get_error_logger()
    return RecentErrorsResponse(
        errors=error_logger.get_recent_errors(limit),
        counts=error_logger.get_error_counts(),
    )

