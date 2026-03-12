"""
SOWTEE Skill Registry
Central registry for managing agent skills.
"""

import structlog
from typing import Any

from .base_skill import BaseSkill, SkillInfo, SkillStatus

logger = structlog.get_logger(__name__)


class SkillRegistry:
    """
    Central registry for agent skills.
    
    Manages skill lifecycle, routing, and status tracking.
    """
    
    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all registered skills."""
        if self._initialized:
            return
            
        for skill_id, skill in self._skills.items():
            try:
                await skill.initialize()
                logger.info("Skill initialized", skill_id=skill_id)
            except Exception as e:
                logger.error("Failed to initialize skill", skill_id=skill_id, error=str(e))
        
        self._initialized = True
    
    def register_skill(self, skill: BaseSkill) -> None:
        """
        Register a skill with the registry.
        
        Args:
            skill: The skill instance to register
        """
        info = skill.get_info()
        if info.skill_id in self._skills:
            logger.warning("Skill already registered, replacing", skill_id=info.skill_id)
        
        self._skills[info.skill_id] = skill
        logger.info("Skill registered", skill_id=info.skill_id, name=info.name)
    
    def get_skill(self, skill_id: str) -> BaseSkill | None:
        """
        Get a skill by ID.
        
        Args:
            skill_id: The skill identifier
            
        Returns:
            The skill instance or None if not found
        """
        return self._skills.get(skill_id)
    
    def list_skills(self) -> list[SkillInfo]:
        """
        List all registered skills.
        
        Returns:
            List of skill information objects
        """
        return [skill.get_info() for skill in self._skills.values()]
    
    def list_active_skills(self) -> list[SkillInfo]:
        """
        List only active skills.
        
        Returns:
            List of active skill information objects
        """
        return [
            info for info in self.list_skills() 
            if info.status == SkillStatus.ACTIVE
        ]
    
    def get_skill_status(self, skill_id: str) -> SkillStatus | None:
        """Get the status of a specific skill."""
        skill = self.get_skill(skill_id)
        if skill:
            return skill.get_info().status
        return None
    
    async def cleanup(self) -> None:
        """Cleanup all skills on shutdown."""
        for skill_id, skill in self._skills.items():
            try:
                await skill.cleanup()
                logger.info("Skill cleaned up", skill_id=skill_id)
            except Exception as e:
                logger.error("Failed to cleanup skill", skill_id=skill_id, error=str(e))


# Singleton instance
_skill_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    """Get the skill registry singleton."""
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = SkillRegistry()
        
        # Auto-register available skills
        _register_default_skills(_skill_registry)
    
    return _skill_registry


def _register_default_skills(registry: SkillRegistry) -> None:
    """Register the default skills."""
    try:
        from .speaking import SpeakingSkill
        registry.register_skill(SpeakingSkill())
        logger.info("Default skills registered")
    except ImportError as e:
        logger.warning("Could not import default skills", error=str(e))
