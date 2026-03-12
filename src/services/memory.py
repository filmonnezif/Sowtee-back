"""
SOWTEE Memory Service
Long-term memory layer using ChromaDB for self-learning personalization.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings

from ..config import get_settings
from ..models import MemoryRecord, VisualContext

logger = structlog.get_logger(__name__)


class MemoryService:
    """
    Self-learning memory service using vector embeddings.
    
    Responsibilities:
    - Store user selections with visual context
    - Retrieve relevant memories based on current scene
    - Personalize predictions through usage patterns
    - Manage memory lifecycle and cleanup
    """
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=self.settings.chroma_persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        self._collection = self._client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"description": "SOWTEE user memory store"}
        )
        logger.info(
            "Memory service initialized",
            collection=self.settings.chroma_collection_name,
            persist_dir=self.settings.chroma_persist_directory
        )
    
    async def store_selection(
        self,
        user_id: str,
        selected_phrase: str,
        visual_context: VisualContext,
        session_id: str | None = None,
    ) -> MemoryRecord:
        """
        Store a user's phrase selection in long-term memory.
        
        This is the LEARN phase of the agentic loop.
        
        Args:
            user_id: Unique user identifier
            selected_phrase: The phrase the user selected
            visual_context: The visual context when selection was made
            session_id: Optional session identifier
            
        Returns:
            The created MemoryRecord
        """
        logger.debug("Storing selection in memory", user_id=user_id, phrase=selected_phrase)
        
        # Create embedding text combining context and phrase
        embedding_text = self._create_embedding_text(visual_context, selected_phrase)
        
        # Check for existing similar memory to update
        existing = await self._find_similar_memory(user_id, selected_phrase, visual_context)
        
        if existing:
            # Update existing memory with incremented count
            return await self._update_memory(existing, visual_context)
        
        # Create new memory record
        record_id = str(uuid4())
        objects_present = [obj.label for obj in visual_context.detected_objects]
        
        metadata = {
            "user_id": user_id,
            "selected_phrase": selected_phrase,
            "objects_present": ",".join(objects_present),
            "environmental_context": visual_context.environmental_context,
            "activity_inference": visual_context.activity_inference,
            "selection_count": 1,
            "session_id": session_id or "",
            "created_at": datetime.utcnow().isoformat(),
            "last_used": datetime.utcnow().isoformat(),
        }
        
        self._collection.add(
            ids=[record_id],
            documents=[embedding_text],
            metadatas=[metadata]
        )
        
        logger.info("Memory stored", record_id=record_id, phrase=selected_phrase)
        
        return MemoryRecord(
            id=record_id,
            user_id=user_id,
            visual_context_summary=visual_context.scene_description,
            selected_phrase=selected_phrase,
            objects_present=objects_present,
            environmental_context=visual_context.environmental_context,
        )
    
    async def retrieve_relevant_memories(
        self,
        user_id: str,
        visual_context: VisualContext,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        """
        Retrieve memories relevant to the current visual context.
        
        This informs the REASON phase of the agentic loop.
        
        Args:
            user_id: User to retrieve memories for
            visual_context: Current visual context
            limit: Max memories to retrieve
            
        Returns:
            List of relevant MemoryRecord objects
        """
        limit = limit or self.settings.memory_retrieval_limit
        
        # Create query from current context
        query_text = self._create_query_text(visual_context)
        
        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=limit,
                where={"user_id": user_id},
            )
            
            return self._parse_query_results(results)
            
        except Exception as e:
            logger.warning("Memory retrieval failed", error=str(e))
            return []
    
    async def get_user_phrase_frequencies(
        self,
        user_id: str,
        limit: int = 20
    ) -> list[tuple[str, int]]:
        """Get most frequently used phrases for a user."""
        try:
            results = self._collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            phrase_counts: dict[str, int] = {}
            for metadata in results.get("metadatas", []):
                phrase = metadata.get("selected_phrase", "")
                count = int(metadata.get("selection_count", 1))
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + count
            
            sorted_phrases = sorted(
                phrase_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return sorted_phrases[:limit]
            
        except Exception as e:
            logger.warning("Failed to get phrase frequencies", error=str(e))
            return []
    
    def _create_embedding_text(
        self,
        visual_context: VisualContext,
        phrase: str
    ) -> str:
        """Create text for embedding that captures context-phrase relationship."""
        objects = ", ".join(obj.label for obj in visual_context.detected_objects)
        return (
            f"Scene: {visual_context.scene_description} | "
            f"Objects: {objects} | "
            f"Location: {visual_context.environmental_context} | "
            f"Activity: {visual_context.activity_inference} | "
            f"Phrase: {phrase}"
        )
    
    def _create_query_text(self, visual_context: VisualContext) -> str:
        """Create query text from visual context."""
        objects = ", ".join(obj.label for obj in visual_context.detected_objects)
        return (
            f"Scene: {visual_context.scene_description} | "
            f"Objects: {objects} | "
            f"Location: {visual_context.environmental_context} | "
            f"Activity: {visual_context.activity_inference}"
        )
    
    async def _find_similar_memory(
        self,
        user_id: str,
        phrase: str,
        visual_context: VisualContext
    ) -> dict[str, Any] | None:
        """Find an existing memory for the same phrase in similar context."""
        try:
            results = self._collection.get(
                where={
                    "$and": [
                        {"user_id": user_id},
                        {"selected_phrase": phrase}
                    ]
                },
                include=["metadatas"]
            )
            
            if results["ids"]:
                # Return first match (could be improved with similarity scoring)
                return {
                    "id": results["ids"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {}
                }
            return None
            
        except Exception:
            return None
    
    async def _update_memory(
        self,
        existing: dict[str, Any],
        visual_context: VisualContext
    ) -> MemoryRecord:
        """Update an existing memory with new usage."""
        record_id = existing["id"]
        metadata = existing["metadata"]
        
        new_count = int(metadata.get("selection_count", 1)) + 1
        
        self._collection.update(
            ids=[record_id],
            metadatas=[{
                **metadata,
                "selection_count": new_count,
                "last_used": datetime.utcnow().isoformat(),
            }]
        )
        
        logger.debug("Memory updated", record_id=record_id, new_count=new_count)
        
        return MemoryRecord(
            id=record_id,
            user_id=metadata.get("user_id", ""),
            visual_context_summary=visual_context.scene_description,
            selected_phrase=metadata.get("selected_phrase", ""),
            objects_present=metadata.get("objects_present", "").split(","),
            environmental_context=metadata.get("environmental_context", ""),
            selection_count=new_count,
        )
    
    def _parse_query_results(self, results: dict[str, Any]) -> list[MemoryRecord]:
        """Parse ChromaDB query results into MemoryRecord objects."""
        records = []
        
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        
        for i, record_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""
            
            objects = metadata.get("objects_present", "")
            objects_list = objects.split(",") if objects else []
            
            records.append(MemoryRecord(
                id=record_id,
                user_id=metadata.get("user_id", ""),
                visual_context_summary=document.split("|")[0].replace("Scene:", "").strip(),
                selected_phrase=metadata.get("selected_phrase", ""),
                objects_present=objects_list,
                environmental_context=metadata.get("environmental_context", ""),
                selection_count=int(metadata.get("selection_count", 1)),
            ))
        
        return records
    
    # ============== Abbreviation History Methods ==============
    
    async def store_abbreviation_selection(
        self,
        user_id: str,
        abbreviation: str,
        selected_expansion: str,
        scene_description: str | None = None,
        conversation_context: str | None = None,
    ) -> dict[str, Any]:
        """
        Store a user's abbreviation expansion selection for learning.
        
        Similar to how intent predictions learn, this allows the abbreviation
        expander to remember what expansions the user prefers in similar contexts.
        
        Args:
            user_id: Unique user identifier
            abbreviation: The abbreviation (e.g., "i w t s")
            selected_expansion: The expansion the user selected
            scene_description: Visual scene context
            conversation_context: Recent conversation history
            
        Returns:
            The stored record info
        """
        logger.debug(
            "Storing abbreviation selection",
            user_id=user_id,
            abbreviation=abbreviation,
            expansion=selected_expansion,
        )
        
        # Create embedding text for similarity search
        embedding_text = self._create_abbreviation_embedding_text(
            abbreviation, selected_expansion, scene_description, conversation_context
        )
        
        # Check for existing similar memory
        existing = await self._find_similar_abbreviation_memory(
            user_id, abbreviation, selected_expansion
        )
        
        if existing:
            # Update existing memory with incremented count
            return await self._update_abbreviation_memory(existing)
        
        # Create new memory record
        record_id = str(uuid4())
        
        metadata = {
            "user_id": user_id,
            "record_type": "abbreviation",  # Distinguish from phrase memories
            "abbreviation": abbreviation,
            "selected_expansion": selected_expansion,
            "scene_description": scene_description or "",
            "conversation_context": (conversation_context or "")[:500],  # Limit size
            "selection_count": 1,
            "created_at": datetime.utcnow().isoformat(),
            "last_used": datetime.utcnow().isoformat(),
        }
        
        self._collection.add(
            ids=[record_id],
            documents=[embedding_text],
            metadatas=[metadata]
        )
        
        logger.info(
            "Abbreviation memory stored",
            record_id=record_id,
            abbreviation=abbreviation,
            expansion=selected_expansion,
        )
        
        return {
            "id": record_id,
            "abbreviation": abbreviation,
            "expansion": selected_expansion,
            "selection_count": 1,
        }
    
    async def retrieve_abbreviation_history(
        self,
        user_id: str,
        abbreviation: str,
        scene_description: str | None = None,
        conversation_context: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant abbreviation expansion history.
        
        Returns past expansions the user selected for similar abbreviations
        in similar contexts, ranked by relevance and frequency.
        
        Args:
            user_id: User to retrieve history for
            abbreviation: Current abbreviation
            scene_description: Current scene
            conversation_context: Current conversation
            limit: Max records to retrieve
            
        Returns:
            List of relevant expansion records with selection counts
        """
        # First, get exact matches for this abbreviation
        exact_matches = await self._get_exact_abbreviation_matches(
            user_id, abbreviation, limit
        )
        
        # If we have enough exact matches, return them sorted by frequency
        if len(exact_matches) >= limit:
            return sorted(
                exact_matches, 
                key=lambda x: x.get("selection_count", 0), 
                reverse=True
            )[:limit]
        
        # Also do semantic search for similar contexts
        if scene_description or conversation_context:
            query_text = self._create_abbreviation_embedding_text(
                abbreviation, "", scene_description, conversation_context
            )
            
            try:
                results = self._collection.query(
                    query_texts=[query_text],
                    n_results=limit,
                    where={
                        "$and": [
                            {"user_id": user_id},
                            {"record_type": "abbreviation"},
                        ]
                    },
                )
                
                semantic_matches = self._parse_abbreviation_results(results)
                
                # Merge exact and semantic matches, prioritizing exact
                seen_expansions = {m["expansion"] for m in exact_matches}
                for match in semantic_matches:
                    if match["expansion"] not in seen_expansions:
                        exact_matches.append(match)
                        seen_expansions.add(match["expansion"])
                
            except Exception as e:
                logger.warning("Abbreviation semantic search failed", error=str(e))
        
        # Sort by selection count
        return sorted(
            exact_matches,
            key=lambda x: x.get("selection_count", 0),
            reverse=True
        )[:limit]
    
    async def _get_exact_abbreviation_matches(
        self,
        user_id: str,
        abbreviation: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get exact matches for an abbreviation."""
        try:
            results = self._collection.get(
                where={
                    "$and": [
                        {"user_id": user_id},
                        {"record_type": "abbreviation"},
                        {"abbreviation": abbreviation},
                    ]
                },
                include=["metadatas"],
            )
            
            matches = []
            for i, record_id in enumerate(results.get("ids", [])):
                metadata = results["metadatas"][i] if results.get("metadatas") else {}
                matches.append({
                    "id": record_id,
                    "abbreviation": metadata.get("abbreviation", ""),
                    "expansion": metadata.get("selected_expansion", ""),
                    "selection_count": int(metadata.get("selection_count", 1)),
                    "scene_description": metadata.get("scene_description", ""),
                    "last_used": metadata.get("last_used", ""),
                })
            
            return matches[:limit]
            
        except Exception as e:
            logger.warning("Failed to get exact abbreviation matches", error=str(e))
            return []
    
    async def _find_similar_abbreviation_memory(
        self,
        user_id: str,
        abbreviation: str,
        expansion: str,
    ) -> dict[str, Any] | None:
        """Find existing memory for same abbreviation-expansion pair."""
        try:
            results = self._collection.get(
                where={
                    "$and": [
                        {"user_id": user_id},
                        {"record_type": "abbreviation"},
                        {"abbreviation": abbreviation},
                        {"selected_expansion": expansion},
                    ]
                },
                include=["metadatas"],
            )
            
            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {},
                }
            return None
            
        except Exception:
            return None
    
    async def _update_abbreviation_memory(
        self,
        existing: dict[str, Any],
    ) -> dict[str, Any]:
        """Update existing abbreviation memory with new usage."""
        record_id = existing["id"]
        metadata = existing["metadata"]
        
        new_count = int(metadata.get("selection_count", 1)) + 1
        
        self._collection.update(
            ids=[record_id],
            metadatas=[{
                **metadata,
                "selection_count": new_count,
                "last_used": datetime.utcnow().isoformat(),
            }]
        )
        
        logger.debug(
            "Abbreviation memory updated",
            record_id=record_id,
            new_count=new_count,
        )
        
        return {
            "id": record_id,
            "abbreviation": metadata.get("abbreviation", ""),
            "expansion": metadata.get("selected_expansion", ""),
            "selection_count": new_count,
        }
    
    def _create_abbreviation_embedding_text(
        self,
        abbreviation: str,
        expansion: str,
        scene_description: str | None,
        conversation_context: str | None,
    ) -> str:
        """Create embedding text for abbreviation memory."""
        parts = [f"Abbreviation: {abbreviation}"]
        if expansion:
            parts.append(f"Expansion: {expansion}")
        if scene_description:
            parts.append(f"Scene: {scene_description}")
        if conversation_context:
            # Take last 200 chars of conversation for embedding
            parts.append(f"Conversation: {conversation_context[-200:]}")
        return " | ".join(parts)
    
    def _parse_abbreviation_results(
        self,
        results: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Parse abbreviation query results."""
        records = []
        
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        for i, record_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Only include abbreviation records
            if metadata.get("record_type") != "abbreviation":
                continue
            
            records.append({
                "id": record_id,
                "abbreviation": metadata.get("abbreviation", ""),
                "expansion": metadata.get("selected_expansion", ""),
                "selection_count": int(metadata.get("selection_count", 1)),
                "scene_description": metadata.get("scene_description", ""),
                "last_used": metadata.get("last_used", ""),
            })
        
        return records


# Singleton instance
_memory_service: MemoryService | None = None


def get_memory_service() -> MemoryService:
    """Get the memory service singleton."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
