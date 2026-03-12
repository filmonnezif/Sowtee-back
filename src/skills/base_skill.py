"""
SOWTEE Base Skill
Abstract base class for all agent skills.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class SkillStatus(str, Enum):
    """Status of a skill."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class SkillInfo:
    """Information about a skill."""
    skill_id: str
    name: str
    description: str
    icon: str  # Emoji or icon name
    status: SkillStatus = SkillStatus.ACTIVE
    version: str = "1.0.0"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "status": self.status.value,
            "version": self.version,
        }


@dataclass
class SkillContext:
    """Context provided to skills for processing."""
    user_id: str
    session_id: str
    scene_description: str | None = None
    scene_image: str | None = None  # Base64 encoded
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    # Last N turns: [{"speaker": "other"|"user", "text": "..."}]
    
    def get_last_heard(self) -> str | None:
        """Get the last thing heard from others."""
        for turn in reversed(self.conversation_history):
            if turn.get("speaker") == "other":
                return turn.get("text")
        return None
    
    def get_last_user_speech(self) -> str | None:
        """Get the last thing the user said."""
        for turn in reversed(self.conversation_history):
            if turn.get("speaker") == "user":
                return turn.get("text")
        return None
    
    def get_recent_context_string(self, max_turns: int = 10) -> str:
        """Get recent conversation as a formatted string with rich context."""
        recent = self.conversation_history[-max_turns:] if self.conversation_history else []
        lines = []
        for i, turn in enumerate(recent):
            speaker = "Other" if turn.get("speaker") == "other" else "User"
            text = turn.get("text", "")
            # Mark the most recent turn for emphasis
            if i == len(recent) - 1:
                lines.append(f"{speaker} (most recent): {text}")
            else:
                lines.append(f"{speaker}: {text}")
        return "\n".join(lines) if lines else "No conversation history"


class BaseSkill(ABC):
    """
    Abstract base class for agent skills.
    
    Each skill must implement:
    - get_info(): Return skill metadata
    - process(): Main skill processing logic
    """
    
    @abstractmethod
    def get_info(self) -> SkillInfo:
        """Return information about this skill."""
        pass
    
    @abstractmethod
    async def process(self, context: SkillContext, **kwargs) -> dict[str, Any]:
        """
        Process a request for this skill.
        
        Args:
            context: The skill context with user/session info
            **kwargs: Skill-specific arguments
            
        Returns:
            Skill-specific response dictionary
        """
        pass
    
    async def initialize(self) -> None:
        """Initialize the skill. Called when skill is registered."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup skill resources. Called on shutdown."""
        pass
