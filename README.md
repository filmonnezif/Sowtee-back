# SOWTEE Backend

The FastAPI backend for SOWTEE - providing the agentic orchestrator, vision analysis, memory services, and surrounding voice transcription.

## Quick Start

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the server
uv run uvicorn src.main:app --reload --port 8000
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Architecture

### Agentic Orchestrator

The core `AgenticOrchestrator` class implements the **Perceive-Reason-Act-Learn** loop:

```python
# Simplified flow
async def process_frame(request):
    # PERCEIVE: Analyze visual scene
    visual_context = await vision_service.analyze_scene(image)
    
    # REASON: Retrieve memories and synthesize intent
    memories = await memory_service.retrieve_relevant_memories(user_id, visual_context)
    
    # ACT: Generate phrase predictions
    predictions = await intent_service.predict_intent(visual_context, memories)
    
    return predictions

# LEARN: Called when user selects a phrase
async def process_feedback(feedback):
    await memory_service.store_selection(user_id, phrase, visual_context)
```

### Services

- **VisionService**: Gemini 1.5 Flash integration for scene understanding
- **MemoryService**: ChromaDB vector store for personalized learning
- **IntentPredictionService**: Combines vision + memory for phrase generation

## Development

```bash
# Run with auto-reload
uv run uvicorn src.main:app --reload

# Type checking
uv run mypy src/

# Linting
uv run ruff check .

# Tests
uv run pytest
```

## Deploy to Railway

This backend is ready for Railway deployment with:
- `Procfile`
- `railway.toml`

### Steps

1. Push the backend repository to GitHub.
2. In Railway, create a new project from that repo.
3. Set the service root directory to `backend` (if using the monorepo).
4. Add required environment variables in Railway:
    - `GROQ_API_KEY` (required)
    - `GEMINI_API_KEY` (required for vision fallback/features)
    - Optional: `STARTUP_KEY_CHECK_STRICT=true` to force strict mode everywhere
    - Optional overrides: `GEMINI_MODEL`, `TOP_K_PHRASES`, `MEMORY_RETRIEVAL_LIMIT`, `CHROMA_PERSIST_DIRECTORY`
5. Deploy.

At startup, the backend validates API keys and logs provider-level diagnostics with masked fingerprints (never full keys). On Railway, strict mode is enabled automatically: if any configured API key is invalid, app startup fails immediately.

Railway will run:
- Build: detected automatically by Nixpacks
- Start: `uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}`
