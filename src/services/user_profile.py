"""
SOWTEE User Profile Service
Persistent user profile storage for agent personalization.
The agent uses profile data to understand WHO the user is and 
adapt predictions, suggestions, and communication style.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any

import structlog

from ..config import get_settings

logger = structlog.get_logger(__name__)

# Profile field definitions with defaults
DEFAULT_PROFILE: dict[str, Any] = {
    "display_name": "",
    "age": None,
    "condition": "",          # ALS, MS, Cerebral Palsy, etc.
    "condition_stage": "",    # Early, Moderate, Advanced
    "primary_language": "en",
    "secondary_language": "",
    "location": "",
    "living_situation": "",   # "With family", "Care home", "Independent with aide"
    "interests": [],          # ["reading", "football", "cooking"]
    "daily_routine": "",      # Free-text: "I wake up at 7am, have breakfast..."
    "communication_style": "casual",  # "formal", "casual", "brief", "detailed"
    "common_needs": [],       # ["water", "medication", "bathroom", "pain relief"]
    "caregiver_name": "",
    "notes": "",              # Anything else the agent should know
    "cloned_voice_id": None,
    "cloned_voice_name": "",
    "created_at": "",
    "updated_at": "",
}


class UserProfileService:
    """
    Persistent user profile storage using JSON files.
    
    Profiles are stored in data/profiles/{user_id}.json and survive restarts.
    The agent accesses profile data to personalize all predictions.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.profiles_dir = Path(settings.chroma_persist_directory).parent / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        logger.info("UserProfileService initialized", profiles_dir=str(self.profiles_dir))

    def _profile_path(self, user_id: str) -> Path:
        """Get the file path for a user's profile."""
        # Sanitize user_id for filesystem safety
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        return self.profiles_dir / f"{safe_id}.json"

    async def get_profile(self, user_id: str) -> dict[str, Any]:
        """
        Get a user's profile. Returns default profile if none exists.
        """
        path = self._profile_path(user_id)
        
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                # Merge with defaults to handle new fields added after creation
                merged = {**DEFAULT_PROFILE, **profile}
                logger.debug("Profile loaded", user_id=user_id)
                return merged
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Failed to load profile", user_id=user_id, error=str(e))
                return {**DEFAULT_PROFILE}
        
        return {**DEFAULT_PROFILE}

    async def save_profile(self, user_id: str, profile_data: dict[str, Any]) -> dict[str, Any]:
        """
        Save or update a user's profile.
        
        Merges with existing data so partial updates work.
        """
        path = self._profile_path(user_id)
        
        # Load existing profile if it exists
        existing = await self.get_profile(user_id)
        
        # Merge: new data overwrites existing
        merged = {**existing, **profile_data}
        
        # Set timestamps
        now = datetime.utcnow().isoformat()
        if not merged.get("created_at"):
            merged["created_at"] = now
        merged["updated_at"] = now
        
        # Write to disk
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            logger.info("Profile saved", user_id=user_id, path=str(path))
        except OSError as e:
            logger.error("Failed to save profile", user_id=user_id, error=str(e))
            raise
        
        return merged

    async def get_profile_context_string(self, user_id: str) -> str:
        """
        Format the user profile into a context string for LLM prompts.
        
        This is how the agent "knows" the user. Returned string is injected
        into system prompts for intent prediction, suggestions, etc.
        """
        profile = await self.get_profile(user_id)
        
        # Only include non-empty fields
        parts: list[str] = []
        
        if profile.get("display_name"):
            parts.append(f"User's name: {profile['display_name']}")
        
        if profile.get("age"):
            parts.append(f"Age: {profile['age']}")
        
        if profile.get("condition"):
            condition_str = profile["condition"]
            if profile.get("condition_stage"):
                condition_str += f" ({profile['condition_stage']} stage)"
            parts.append(f"Medical condition: {condition_str}")
        
        if profile.get("primary_language"):
            lang_str = f"Primary language: {profile['primary_language']}"
            if profile.get("secondary_language"):
                lang_str += f", also speaks {profile['secondary_language']}"
            parts.append(lang_str)
        
        if profile.get("location"):
            parts.append(f"Location: {profile['location']}")
        
        if profile.get("living_situation"):
            parts.append(f"Living situation: {profile['living_situation']}")
        
        if profile.get("interests"):
            interests = profile["interests"]
            if isinstance(interests, list) and interests:
                parts.append(f"Interests: {', '.join(interests)}")
        
        if profile.get("daily_routine"):
            parts.append(f"Daily routine: {profile['daily_routine']}")
        
        if profile.get("communication_style"):
            parts.append(f"Communication style preference: {profile['communication_style']}")
        
        if profile.get("common_needs"):
            needs = profile["common_needs"]
            if isinstance(needs, list) and needs:
                parts.append(f"Common needs: {', '.join(needs)}")
        
        if profile.get("caregiver_name"):
            parts.append(f"Primary caregiver: {profile['caregiver_name']}")
        
        if profile.get("notes"):
            parts.append(f"Additional context: {profile['notes']}")
        
        if not parts:
            return ""
        
        return "USER PROFILE:\n" + "\n".join(f"- {p}" for p in parts)


# Singleton instance
_profile_service: UserProfileService | None = None


def get_profile_service() -> UserProfileService:
    """Get the user profile service singleton."""
    global _profile_service
    if _profile_service is None:
        _profile_service = UserProfileService()
    return _profile_service
