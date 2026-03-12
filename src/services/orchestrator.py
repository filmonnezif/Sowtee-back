"""
SOWTEE Agentic Orchestrator
The core reasoning engine implementing the Perceive-Reason-Act-Learn loop.
Now with self-improving strategy selection, agent tools, and learning tracking.
"""

import time
from datetime import datetime
from uuid import uuid4

import structlog

from ..config import get_settings
from ..models import (
    AgentPhase,
    AgentState,
    ContextAnalysis,
    PhraseCandidate,
    PredictionMode,
    PredictionRequest,
    PredictionResponse,
    UserFeedback,
    VisualContext,
)
from .intent import get_intent_service, IntentPredictionService
from .memory import get_memory_service, MemoryService
from .vision import get_vision_service, VisionService

logger = structlog.get_logger(__name__)


class AgenticOrchestrator:
    """
    The central orchestrator implementing the agentic cognitive loop.
    
    The loop follows four phases:
    1. PERCEIVE: Analyze visual input to understand the user's environment
    2. REASON: Synthesize visual context with memory to predict intent
       - Select best learning strategy (epsilon-greedy)
       - Execute relevant agent tools
    3. ACT: Present predicted phrases to the user
    4. LEARN: Store user selections AND track learning metrics
    
    Self-improvement mechanisms:
    - Strategy selection adapts based on measured success rates
    - Agent tools are chosen based on context and past effectiveness
    - Learning tracker quantifies improvement over time
    """
    
    def __init__(
        self,
        vision_service: VisionService | None = None,
        memory_service: MemoryService | None = None,
        intent_service: IntentPredictionService | None = None,
    ) -> None:
        self.settings = get_settings()
        self._vision = vision_service or get_vision_service()
        self._memory = memory_service or get_memory_service()
        self._intent = intent_service or get_intent_service()
        
        # Active sessions for state management
        self._sessions: dict[str, AgentState] = {}
        
        # Track per-session strategy and tools for the LEARN phase
        self._session_meta: dict[str, dict] = {}
        
        logger.info("Agentic Orchestrator initialized (self-learning enabled)")
    
    async def process_frame(self, request: PredictionRequest) -> PredictionResponse:
        """
        Process a camera frame through the full agentic loop.
        
        This is the main entry point for the frontend. It orchestrates:
        1. Strategy selection (self-improving)
        2. Visual perception
        3. Agent tool execution
        4. Intent prediction with strategy-adapted prompts
        5. Learning metric recording
        
        Args:
            request: PredictionRequest containing image and user context
            
        Returns:
            PredictionResponse with predicted phrases
        """
        start_time = time.time()
        
        # Initialize or retrieve session state
        session_id = request.session_id or str(uuid4())
        state = self._get_or_create_state(session_id, request.user_id)
        
        reasoning_trace: list[str] = []
        strategy_used = "default"
        tools_used: list[str] = []
        
        try:
            # ============== AGENTIC CYCLE HEADER ==============
            logger.info(
                "\n"
                "    ╔══════════════════════════════════════════════════════╗\n"
                "    ║       🔄 AGENTIC CYCLE — Perceive → Reason → Act   ║\n"
                "    ╠══════════════════════════════════════════════════════╣\n"
                f"    ║  User: {request.user_id[:30]:<30}         ║\n"
                f"    ║  Session: {session_id[:30]:<30}      ║\n"
                f"    ║  Mode: {request.mode.value:<30}           ║\n"
                "    ╚══════════════════════════════════════════════════════╝",
            )
            
            # ============== PHASE 1: PERCEIVE ==============
            state.current_phase = AgentPhase.PERCEIVE
            reasoning_trace.append("🔍 PERCEIVE: Analyzing visual scene...")
            
            visual_context = await self._perceive(request.image_base64)
            state.visual_context = visual_context
            
            reasoning_trace.append(
                f"  → Detected {len(visual_context.detected_objects)} objects: "
                f"{', '.join(o.label for o in visual_context.detected_objects[:5])}"
            )
            reasoning_trace.append(f"  → Environment: {visual_context.environmental_context}")
            reasoning_trace.append(f"  → Activity: {visual_context.activity_inference}")
            
            # ============== PHASE 2: REASON ==============
            # Skip intent prediction if in vision_only or abbreviation mode
            skip_intent = request.mode in (PredictionMode.VISION_ONLY, PredictionMode.ABBREVIATION)
            
            memories = []
            candidates = []
            strategy_context = ""
            
            if skip_intent:
                state.current_phase = AgentPhase.REASON
                reasoning_trace.append("🧠 REASON: Skipped (optimization mode - user is typing)")
                reasoning_trace.append(f"  → Mode: {request.mode.value}")
            else:
                state.current_phase = AgentPhase.REASON
                reasoning_trace.append("🧠 REASON: Retrieving memories and synthesizing intent...")
                
                # ── Step 2a: Retrieve memories ──
                memories = await self._memory.retrieve_relevant_memories(
                    request.user_id,
                    visual_context,
                )
                state.retrieved_memories = memories
                
                if memories:
                    reasoning_trace.append(f"  → Found {len(memories)} relevant memories")
                    top_memory = memories[0] if memories else None
                    if top_memory:
                        reasoning_trace.append(
                            f"  → Most relevant past phrase: \"{top_memory.selected_phrase}\" "
                            f"(used {top_memory.selection_count}x)"
                        )
                else:
                    reasoning_trace.append("  → No relevant memories found (new context)")
                
                # ── Step 2b: Select learning strategy ──
                try:
                    from .strategy_manager import get_strategy_manager
                    strategy_mgr = get_strategy_manager()
                    strategy = await strategy_mgr.select_strategy(
                        user_id=request.user_id,
                        has_memories=len(memories) > 0,
                        has_scene=bool(visual_context.scene_description),
                        has_conversation=bool(request.additional_context),
                    )
                    strategy_used = strategy["id"]
                    strategy_context = strategy.get("prompt_modifier", "")
                    
                    reasoning_trace.append(
                        f"  🎯 Strategy: {strategy['name']} ({strategy['selection_reason']})"
                    )
                    reasoning_trace.append(f"  → {strategy['description']}")
                    tools_used.append("strategy_selection")
                except Exception as e:
                    logger.warning("Strategy selection failed, using default", error=str(e))
                    reasoning_trace.append("  ⚠️ Strategy selection failed, using default")
                
                # ── Step 2c: Execute agent tools ──
                tool_context = await self._execute_tools(
                    request.user_id, visual_context, request.additional_context, reasoning_trace
                )
                tools_used.extend(tool_context.get("tools_executed", []))
                
                # ── Step 2d: Build enriched context ──
                from .user_profile import get_profile_service
                profile_service = get_profile_service()
                profile_context = await profile_service.get_profile_context_string(request.user_id)
                
                enriched_context = request.additional_context or ""
                
                # Inject profile context
                if profile_context:
                    enriched_context = f"{profile_context}\n\n{enriched_context}" if enriched_context else profile_context
                
                # Inject strategy modifier
                if strategy_context:
                    enriched_context = f"STRATEGY GUIDANCE: {strategy_context}\n\n{enriched_context}"
                
                # Inject tool contributions
                tool_contributions = tool_context.get("contributions", "")
                if tool_contributions:
                    enriched_context = f"{enriched_context}\n\n{tool_contributions}"
                
                # ── Step 2e: Generate intent predictions ──
                candidates = await self._intent.predict_intent(
                    visual_context,
                    memories,
                    enriched_context,
                )
                state.candidate_phrases = candidates
                
                reasoning_trace.append(f"  → Generated {len(candidates)} phrase predictions")
            
            # ============== PHASE 3: ACT ==============
            state.current_phase = AgentPhase.ACT
            reasoning_trace.append("🎯 ACT: Preparing response for user...")
            
            for i, candidate in enumerate(candidates, 1):
                reasoning_trace.append(
                    f"  → Option {i}: \"{candidate.phrase}\" "
                    f"(confidence: {candidate.confidence:.0%}, source: {candidate.source})"
                )
            
            # Add learning info to reasoning trace
            if strategy_used != "default":
                reasoning_trace.append(f"  📊 Strategy used: {strategy_used}")
            if tools_used:
                reasoning_trace.append(f"  🔧 Tools used: {', '.join(tools_used)}")
            
            # Store reasoning trace
            state.reasoning_trace = reasoning_trace
            state.cycle_count += 1
            state.last_updated = datetime.utcnow()
            
            # Store session metadata for the LEARN phase
            processing_time_ms = (time.time() - start_time) * 1000
            self._session_meta[session_id] = {
                "strategy_used": strategy_used,
                "tools_used": tools_used,
                "predicted_phrases": [c.phrase for c in candidates],
                "processing_time_ms": processing_time_ms,
            }
            
            logger.info(
                "Frame processed",
                session_id=session_id,
                cycle=state.cycle_count,
                candidates=len(candidates),
                strategy=strategy_used,
                tools=tools_used,
                processing_time_ms=round(processing_time_ms, 2),
            )
            
            return PredictionResponse(
                session_id=session_id,
                phrases=candidates,
                context_analysis=ContextAnalysis(
                    visual_context=visual_context,
                    retrieved_memories=memories,
                    reasoning_trace=reasoning_trace,
                    processing_time_ms=processing_time_ms,
                ),
                agent_phase=AgentPhase.ACT,
                processing_time_ms=processing_time_ms,
            )
            
        except Exception as e:
            logger.error("Frame processing failed", error=str(e), session_id=session_id)
            
            # Return graceful fallback
            return PredictionResponse(
                session_id=session_id,
                phrases=self._get_fallback_phrases(),
                context_analysis=ContextAnalysis(
                    visual_context=state.visual_context or VisualContext(
                        scene_description="Unable to analyze scene",
                        detected_objects=[],
                    ),
                    retrieved_memories=[],
                    reasoning_trace=[f"⚠️ Error: {str(e)}"],
                    processing_time_ms=(time.time() - start_time) * 1000,
                ),
                agent_phase=AgentPhase.ACT,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    async def process_feedback(self, feedback: UserFeedback) -> None:
        """
        Process user feedback (phrase selection) for learning.
        
        This is the LEARN phase of the agentic loop.
        Now also records learning metrics for self-improvement tracking.
        
        Args:
            feedback: UserFeedback containing selected phrase and context
        """
        logger.debug("Processing feedback", session_id=feedback.session_id)
        
        # Get session state for context
        state = self._sessions.get(feedback.session_id)
        
        if state and state.visual_context:
            visual_context = state.visual_context
        else:
            # Reconstruct minimal context from feedback
            from ..models import DetectedObject
            visual_context = VisualContext(
                scene_description=feedback.visual_context_summary,
                detected_objects=[
                    DetectedObject(label=obj, confidence=0.8)
                    for obj in feedback.objects_present
                ],
            )
        
        # Store in memory for future learning
        phrase = feedback.custom_phrase or feedback.selected_phrase
        
        await self._memory.store_selection(
            user_id=feedback.user_id,
            selected_phrase=phrase,
            visual_context=visual_context,
            session_id=feedback.session_id,
        )
        
        # ── SELF-IMPROVEMENT: Record learning metrics ──
        try:
            from .learning_tracker import get_learning_tracker
            tracker = get_learning_tracker()
            
            # Get session metadata (strategy, tools, predictions)
            meta = self._session_meta.get(feedback.session_id, {})
            predicted_phrases = meta.get("predicted_phrases", [])
            strategy_used = meta.get("strategy_used", "default")
            tools_used = meta.get("tools_used", [])
            response_time = meta.get("processing_time_ms", 0.0)
            
            match = phrase in predicted_phrases if phrase else False
            match_icon = "✅" if match else "❌"
            
            logger.info(
                "\n"
                "    ╔══════════════════════════════════════════════════════╗\n"
                "    ║       📖 LEARN PHASE — Self-Improvement             ║\n"
                "    ╠══════════════════════════════════════════════════════╣\n"
                f"    ║  {match_icon} Prediction {'HIT' if match else 'MISS'}\n"
                f"    ║  User selected: \"{(phrase or 'dismissed')[:40]}\"\n"
                f"    ║  Strategy was: {strategy_used}\n"
                f"    ║  Tools used: {', '.join(tools_used) or 'none'}\n"
                f"    ║  From suggestions: {feedback.was_from_predictions}\n"
                "    ║  → Recording for future improvement...\n"
                "    ╚══════════════════════════════════════════════════════╝",
            )
            
            await tracker.record_prediction(
                user_id=feedback.user_id,
                predicted_phrases=predicted_phrases,
                selected_phrase=phrase,
                strategy_used=strategy_used,
                tools_used=tools_used,
                response_time_ms=response_time,
                was_from_predictions=feedback.was_from_predictions,
            )
        except Exception as e:
            logger.warning("Failed to record learning metrics", error=str(e))
        
        logger.info(
            "Feedback stored",
            session_id=feedback.session_id,
            phrase=phrase[:50],
            was_prediction=feedback.was_from_predictions,
        )
    
    async def _execute_tools(
        self,
        user_id: str,
        visual_context: VisualContext,
        additional_context: str | None,
        reasoning_trace: list[str],
    ) -> dict:
        """
        Execute relevant agent tools and collect their contributions.
        
        Returns dict with:
        - tools_executed: list of tool names
        - contributions: combined context string
        """
        tools_executed = []
        contributions = []
        
        try:
            from .agent_tools import get_available_tools
            tools = get_available_tools()
            
            # Execute profile lookup (always useful)
            profile_tool = tools.get("profile_lookup")
            if profile_tool:
                result = await profile_tool.execute(user_id=user_id)
                if result.success and result.context_contribution:
                    tools_executed.append(result.tool_name)
                    # Profile is already injected separately, skip adding to contributions
            
            # Execute frequency analysis (useful for returning users)
            freq_tool = tools.get("frequency_analysis")
            if freq_tool:
                result = await freq_tool.execute(user_id=user_id)
                if result.success and result.context_contribution:
                    tools_executed.append(result.tool_name)
                    contributions.append(result.context_contribution)
            
            # Execute vocabulary expansion tracking
            vocab_tool = tools.get("vocabulary_expansion")
            if vocab_tool:
                result = await vocab_tool.execute(user_id=user_id)
                if result.success:
                    tools_executed.append(result.tool_name)
            
            if tools_executed:
                reasoning_trace.append(f"  🔧 Agent tools executed: {', '.join(tools_executed)}")
                
        except Exception as e:
            logger.warning("Tool execution failed", error=str(e))
            reasoning_trace.append(f"  ⚠️ Some agent tools failed: {str(e)}")
        
        return {
            "tools_executed": tools_executed,
            "contributions": "\n\n".join(contributions) if contributions else "",
        }
    
    async def _perceive(self, image_base64: str) -> VisualContext:
        """Execute the PERCEIVE phase using vision service."""
        return await self._vision.analyze_scene(image_base64)
    
    def _get_or_create_state(self, session_id: str, user_id: str) -> AgentState:
        """Get existing session state or create new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = AgentState(
                session_id=session_id,
                user_id=user_id,
            )
            logger.debug("New session created", session_id=session_id)
        
        return self._sessions[session_id]
    
    def _get_fallback_phrases(self) -> list[PhraseCandidate]:
        """Get fallback phrases when processing fails."""
        return [
            PhraseCandidate(
                phrase="I need help",
                phrase_arabic="أحتاج مساعدة",
                confidence=0.9,
                reasoning="Fallback - general assistance",
                source="fallback",
            ),
            PhraseCandidate(
                phrase="Yes",
                phrase_arabic="نعم",
                confidence=0.8,
                reasoning="Fallback - simple affirmation",
                source="fallback",
            ),
            PhraseCandidate(
                phrase="No",
                phrase_arabic="لا",
                confidence=0.8,
                reasoning="Fallback - simple negation",
                source="fallback",
            ),
        ]
    
    def get_session_state(self, session_id: str) -> AgentState | None:
        """Get the current state of a session."""
        return self._sessions.get(session_id)
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session's state."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            # Also clear session metadata
            self._session_meta.pop(session_id, None)
            return True
        return False


# Singleton instance
_orchestrator: AgenticOrchestrator | None = None


def get_orchestrator() -> AgenticOrchestrator:
    """Get the agentic orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgenticOrchestrator()
    return _orchestrator
