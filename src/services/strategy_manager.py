"""
SOWTEE Strategy Manager
Manages and selects learning strategies using epsilon-greedy exploration.
The agent tries different prediction strategies and converges on what works
best for each user over time.
"""

import random
from datetime import datetime
from typing import Any

import structlog

from .learning_tracker import get_learning_tracker

logger = structlog.get_logger(__name__)


# Strategy definitions
STRATEGIES: dict[str, dict[str, Any]] = {
    "memory_first": {
        "name": "Memory First",
        "description": "Prioritize vector memory matches over LLM reasoning",
        "prompt_modifier": (
            "Focus heavily on the user's past selections and patterns. "
            "Prefer phrases the user has used before in similar contexts. "
            "Only generate new phrases if no strong memory matches exist."
        ),
        "memory_weight": 0.8,
        "llm_weight": 0.2,
        "icon": "🧠",
    },
    "llm_reasoning": {
        "name": "LLM Reasoning",
        "description": "Lean heavily on LLM with rich context for novel predictions",
        "prompt_modifier": (
            "Think carefully about what the user most likely wants to communicate "
            "given the current situation. Be creative and consider non-obvious intents. "
            "Generate diverse, contextually appropriate phrases."
        ),
        "memory_weight": 0.2,
        "llm_weight": 0.8,
        "icon": "🤖",
    },
    "hybrid_weighted": {
        "name": "Hybrid Weighted",
        "description": "Dynamically balance memory and LLM based on context richness",
        "prompt_modifier": (
            "Balance between the user's known preferences and contextually appropriate "
            "new suggestions. Use memories to guide but not limit your reasoning."
        ),
        "memory_weight": 0.5,
        "llm_weight": 0.5,
        "icon": "⚖️",
    },
    "frequency_boost": {
        "name": "Frequency Boost",
        "description": "Boost the user's most frequently used phrases",
        "prompt_modifier": (
            "The user tends to repeat certain phrases frequently. "
            "Strongly prioritize their most-used phrases, especially for routine "
            "interactions. Include their top phrases whenever contextually relevant."
        ),
        "memory_weight": 0.7,
        "llm_weight": 0.3,
        "icon": "📊",
    },
    "context_heavy": {
        "name": "Context Heavy",
        "description": "Maximize visual and conversation context for rich predictions",
        "prompt_modifier": (
            "Pay extremely close attention to the visual scene and conversation history. "
            "Generate phrases that are highly specific to what's happening RIGHT NOW. "
            "Reference visible objects, people, and the current activity directly."
        ),
        "memory_weight": 0.3,
        "llm_weight": 0.7,
        "icon": "👁️",
    },
}


class StrategyManager:
    """
    Epsilon-greedy strategy selector.

    Most of the time, picks the best-performing strategy.
    Occasionally explores other strategies to discover improvements.
    """

    # Exploration rate: 20% of the time, try a random strategy
    EPSILON = 0.20
    # Minimum attempts before a strategy's score is trusted
    MIN_ATTEMPTS = 3

    def __init__(self) -> None:
        self._tracker = get_learning_tracker()
        logger.info(
            "StrategyManager initialized",
            strategies=list(STRATEGIES.keys()),
            epsilon=self.EPSILON,
        )

    async def select_strategy(
        self,
        user_id: str,
        has_memories: bool = False,
        has_scene: bool = False,
        has_conversation: bool = False,
    ) -> dict[str, Any]:
        """
        Select the best strategy for this interaction.

        Uses epsilon-greedy: usually picks the best, sometimes explores.

        Returns:
            Dict with strategy info including name, prompt_modifier, weights
        """
        stats = await self._tracker.get_strategy_stats(user_id)

        # Decide: exploit or explore?
        if random.random() < self.EPSILON:
            # EXPLORE: pick a random strategy
            strategy_id = random.choice(list(STRATEGIES.keys()))
            selection_reason = "exploration"
            strat = STRATEGIES[strategy_id]
            logger.info(
                "\n"
                "    ┌──────────────────────────────────────────────┐\n"
                f"    │  🎲 STRATEGY: EXPLORING                      \n"
                "    ├──────────────────────────────────────────────┤\n"
                f"    │  {strat['icon']} {strat['name']}\n"
                f"    │  → {strat['description'][:60]}\n"
                f"    │  Reason: trying something new (ε={self.EPSILON})\n"
                "    └──────────────────────────────────────────────┘",
            )
        else:
            # EXPLOIT: pick the best-performing strategy
            strategy_id = self._pick_best(stats, has_memories, has_scene, has_conversation)
            selection_reason = "exploitation"
            strat = STRATEGIES[strategy_id]
            best_rate = stats.get(strategy_id, {}).get("success_rate", 0)
            logger.info(
                "\n"
                "    ┌──────────────────────────────────────────────┐\n"
                f"    │  🏆 STRATEGY: BEST PICK                      \n"
                "    ├──────────────────────────────────────────────┤\n"
                f"    │  {strat['icon']} {strat['name']}\n"
                f"    │  → {strat['description'][:60]}\n"
                f"    │  Success rate: {best_rate:.1%}\n"
                "    └──────────────────────────────────────────────┘",
            )

        strategy = STRATEGIES[strategy_id].copy()
        strategy["id"] = strategy_id
        strategy["selection_reason"] = selection_reason

        return strategy

    async def record_outcome(
        self,
        user_id: str,
        strategy_id: str,
        was_successful: bool,
    ) -> None:
        """
        Record whether a strategy led to a successful prediction.
        This is handled by the learning tracker's record_prediction.
        """
        # The learning tracker already records strategy outcomes
        # This method exists for explicit strategy-only recording
        logger.debug(
            "Strategy outcome",
            user_id=user_id,
            strategy=strategy_id,
            success=was_successful,
        )

    async def get_all_strategies(self) -> dict[str, dict[str, Any]]:
        """Get all available strategies with their descriptions."""
        return {
            sid: {
                "name": s["name"],
                "description": s["description"],
                "icon": s["icon"],
            }
            for sid, s in STRATEGIES.items()
        }

    def _pick_best(
        self,
        stats: dict[str, Any],
        has_memories: bool,
        has_scene: bool,
        has_conversation: bool,
    ) -> str:
        """Pick the best strategy based on historical performance and context."""
        best_id = "hybrid_weighted"  # Default
        best_score = -1.0

        for strategy_id in STRATEGIES:
            s = stats.get(strategy_id, {})
            attempts = s.get("attempts", 0)
            success_rate = s.get("success_rate", 0.0)

            if attempts < self.MIN_ATTEMPTS:
                # Not enough data — give it a default score to encourage trying
                score = 0.5
            else:
                score = success_rate

            # Context-based bonus
            if strategy_id == "memory_first" and has_memories:
                score += 0.1
            elif strategy_id == "context_heavy" and has_scene:
                score += 0.1
            elif strategy_id == "llm_reasoning" and not has_memories:
                score += 0.1
            elif strategy_id == "frequency_boost" and has_memories:
                score += 0.05

            if score > best_score:
                best_score = score
                best_id = strategy_id

        return best_id


# Singleton
_strategy_manager: StrategyManager | None = None


def get_strategy_manager() -> StrategyManager:
    """Get the strategy manager singleton."""
    global _strategy_manager
    if _strategy_manager is None:
        _strategy_manager = StrategyManager()
    return _strategy_manager
