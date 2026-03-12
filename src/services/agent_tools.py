"""
SOWTEE Agent Tools
A toolbox of capabilities the agent can autonomously choose from.
Each tool is a callable that provides specific information or analysis
to improve prediction quality.
"""

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ToolResult:
    """Result from executing an agent tool."""
    tool_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    context_contribution: str = ""  # Text to inject into LLM prompt
    execution_time_ms: float = 0.0


class AgentTool:
    """Base class for agent tools."""

    name: str = "base_tool"
    description: str = "Base agent tool"
    icon: str = "🔧"

    async def execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
        }


class MemorySearchTool(AgentTool):
    """Search vector memory for relevant past interactions."""

    name = "memory_search"
    description = "Search long-term memory for phrases the user has used in similar contexts"
    icon = "🧠"

    async def execute(self, **kwargs: Any) -> ToolResult:
        import time
        start = time.time()

        user_id = kwargs.get("user_id", "")
        visual_context = kwargs.get("visual_context")

        if not user_id or not visual_context:
            return ToolResult(
                tool_name=self.name,
                success=False,
                context_contribution="",
            )

        try:
            from .memory import get_memory_service
            memory = get_memory_service()
            memories = await memory.retrieve_relevant_memories(user_id, visual_context)

            if not memories:
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    data={"memories_found": 0},
                    context_contribution="No relevant memories found for this context.",
                    execution_time_ms=(time.time() - start) * 1000,
                )

            # Build context string from memories
            memory_lines = []
            for m in memories[:5]:
                count = m.selection_count
                memory_lines.append(
                    f'- "{m.selected_phrase}" (used {count}x, context: {m.visual_context_summary[:60]})'
                )

            contribution = (
                "USER'S PAST SELECTIONS IN SIMILAR CONTEXTS:\n"
                + "\n".join(memory_lines)
            )

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "memories_found": len(memories),
                    "top_phrases": [m.selected_phrase for m in memories[:5]],
                },
                context_contribution=contribution,
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error("MemorySearchTool failed", error=str(e))
            return ToolResult(tool_name=self.name, success=False)


class ProfileLookupTool(AgentTool):
    """Read user profile for personalization context."""

    name = "profile_lookup"
    description = "Look up the user's profile for personalization (name, condition, interests, etc.)"
    icon = "👤"

    async def execute(self, **kwargs: Any) -> ToolResult:
        import time
        start = time.time()

        user_id = kwargs.get("user_id", "")
        if not user_id:
            return ToolResult(tool_name=self.name, success=False)

        try:
            from .user_profile import get_profile_service
            profile_service = get_profile_service()
            context_string = await profile_service.get_profile_context_string(user_id)

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"has_profile": bool(context_string)},
                context_contribution=context_string or "No user profile configured.",
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error("ProfileLookupTool failed", error=str(e))
            return ToolResult(tool_name=self.name, success=False)


class VisionAnalysisTool(AgentTool):
    """Analyze current visual scene."""

    name = "vision_analysis"
    description = "Analyze the camera feed to understand the current visual scene"
    icon = "👁️"

    async def execute(self, **kwargs: Any) -> ToolResult:
        import time
        start = time.time()

        image_base64 = kwargs.get("image_base64")
        if not image_base64:
            return ToolResult(tool_name=self.name, success=False)

        try:
            from .vision import get_vision_service
            vision = get_vision_service()
            visual_context = await vision.analyze_frame(image_base64)

            contribution = (
                f"CURRENT SCENE: {visual_context.scene_description}\n"
                f"Setting: {visual_context.environmental_context}\n"
                f"Activity: {visual_context.activity_inference}"
            )

            objects = [obj.label for obj in visual_context.detected_objects]
            if objects:
                contribution += f"\nVisible objects: {', '.join(objects)}"

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "scene": visual_context.scene_description,
                    "objects": objects,
                    "environment": visual_context.environmental_context,
                },
                context_contribution=contribution,
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error("VisionAnalysisTool failed", error=str(e))
            return ToolResult(tool_name=self.name, success=False)


class ConversationContextTool(AgentTool):
    """Build context from conversation history."""

    name = "conversation_context"
    description = "Analyze the conversation history to understand dialogue flow"
    icon = "💬"

    async def execute(self, **kwargs: Any) -> ToolResult:
        import time
        start = time.time()

        conversation_history = kwargs.get("conversation_history", [])
        if not conversation_history:
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"turns": 0},
                context_contribution="No conversation history available.",
                execution_time_ms=(time.time() - start) * 1000,
            )

        lines = []
        for turn in conversation_history[-10:]:
            speaker = "Other person" if turn.get("speaker") == "other" else "User"
            lines.append(f"{speaker}: {turn.get('text', '')}")

        contribution = "RECENT CONVERSATION:\n" + "\n".join(lines)

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"turns": len(conversation_history)},
            context_contribution=contribution,
            execution_time_ms=(time.time() - start) * 1000,
        )


class FrequencyAnalysisTool(AgentTool):
    """Analyze phrase frequency patterns."""

    name = "frequency_analysis"
    description = "Analyze the user's most frequently used phrases to prioritize common needs"
    icon = "📊"

    async def execute(self, **kwargs: Any) -> ToolResult:
        import time
        start = time.time()

        user_id = kwargs.get("user_id", "")
        if not user_id:
            return ToolResult(tool_name=self.name, success=False)

        try:
            from .memory import get_memory_service
            memory = get_memory_service()
            frequencies = await memory.get_user_phrase_frequencies(user_id, limit=10)

            if not frequencies:
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    data={"phrases_found": 0},
                    context_contribution="No phrase history yet.",
                    execution_time_ms=(time.time() - start) * 1000,
                )

            freq_lines = [f'- "{phrase}" (used {count}x)' for phrase, count in frequencies]
            contribution = "USER'S MOST USED PHRASES:\n" + "\n".join(freq_lines)

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "phrases_found": len(frequencies),
                    "top_phrases": [p for p, _ in frequencies[:5]],
                },
                context_contribution=contribution,
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error("FrequencyAnalysisTool failed", error=str(e))
            return ToolResult(tool_name=self.name, success=False)


class VocabularyExpansionTool(AgentTool):
    """Detect and learn new vocabulary from user input."""

    name = "vocabulary_expansion"
    description = "Track and expand the user's learned vocabulary over time"
    icon = "📚"

    async def execute(self, **kwargs: Any) -> ToolResult:
        import time
        start = time.time()

        user_id = kwargs.get("user_id", "")
        if not user_id:
            return ToolResult(tool_name=self.name, success=False)

        try:
            tracker = get_learning_tracker_instance()
            metrics = await tracker.get_metrics(user_id)

            vocab_size = metrics.get("vocabulary_size", 0)
            contribution = f"User's vocabulary size: {vocab_size} learned phrases."

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"vocabulary_size": vocab_size},
                context_contribution=contribution,
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error("VocabularyExpansionTool failed", error=str(e))
            return ToolResult(tool_name=self.name, success=False)


class StrategySelectionTool(AgentTool):
    """Select the best prediction strategy based on performance data."""

    name = "strategy_selection"
    description = "Choose the best prediction strategy based on what has worked for this user"
    icon = "🎯"

    async def execute(self, **kwargs: Any) -> ToolResult:
        import time
        start = time.time()

        user_id = kwargs.get("user_id", "")
        has_memories = kwargs.get("has_memories", False)
        has_scene = kwargs.get("has_scene", False)
        has_conversation = kwargs.get("has_conversation", False)

        try:
            from .strategy_manager import get_strategy_manager
            manager = get_strategy_manager()
            strategy = await manager.select_strategy(
                user_id,
                has_memories=has_memories,
                has_scene=has_scene,
                has_conversation=has_conversation,
            )

            contribution = (
                f"SELECTED STRATEGY: {strategy['name']} ({strategy['selection_reason']})\n"
                f"Approach: {strategy['prompt_modifier']}"
            )

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "strategy_id": strategy["id"],
                    "strategy_name": strategy["name"],
                    "selection_reason": strategy["selection_reason"],
                    "memory_weight": strategy["memory_weight"],
                    "llm_weight": strategy["llm_weight"],
                },
                context_contribution=contribution,
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error("StrategySelectionTool failed", error=str(e))
            return ToolResult(tool_name=self.name, success=False)


def get_learning_tracker_instance():
    """Helper to avoid circular import."""
    from .learning_tracker import get_learning_tracker
    return get_learning_tracker()


# ── Tool Registry ─────────────────────────────────────────────

# All available tools
ALL_TOOLS: dict[str, AgentTool] = {
    "memory_search": MemorySearchTool(),
    "profile_lookup": ProfileLookupTool(),
    "vision_analysis": VisionAnalysisTool(),
    "conversation_context": ConversationContextTool(),
    "frequency_analysis": FrequencyAnalysisTool(),
    "vocabulary_expansion": VocabularyExpansionTool(),
    "strategy_selection": StrategySelectionTool(),
}


def get_available_tools() -> dict[str, AgentTool]:
    """Get all available agent tools."""
    return ALL_TOOLS


def list_tools_info() -> list[dict[str, str]]:
    """Get info about all tools for display."""
    return [tool.to_dict() for tool in ALL_TOOLS.values()]
