"""
Agent Version Manager

This module provides version control for Azure AI agents, tracking configuration changes
and enabling rollback to previous versions.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from copy import deepcopy


class AgentVersionManager:
    """Manager for agent version control"""

    def __init__(self, agent_manager):
        """
        Initialize the AgentVersionManager

        Args:
            agent_manager: AgentManager instance for agent operations
        """
        self.agent_manager = agent_manager
        self.db = agent_manager.db

    def create_version_snapshot(
        self,
        agent_data: Dict[str, Any],
        change_description: str = "",
        changed_by: str = "system"
    ) -> Dict[str, Any]:
        """
        Create a version snapshot from current agent data

        Args:
            agent_data: Current agent data from database
            change_description: Description of what changed
            changed_by: Who made the change

        Returns:
            Version snapshot dict
        """
        version = {
            "versionNumber": agent_data.get("currentVersion", 0),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "changeDescription": change_description,
            "changedBy": changed_by,
            "snapshot": {
                "name": agent_data.get("name"),
                "description": agent_data.get("description"),
                "instruction": agent_data.get("instruction"),
                "category": agent_data.get("category"),
                "status": agent_data.get("status"),
                "avatarUrl": agent_data.get("avatarUrl"),
                "functionList": deepcopy(agent_data.get("functionList", [])),
                "knowledgeBase": agent_data.get("knowledgeBase", ""),
                "samplePrompts": deepcopy(agent_data.get("samplePrompts", [])),
                "scenarios": deepcopy(agent_data.get("scenarios", [])),
                "maintainers": deepcopy(agent_data.get("maintainers", []))
            }
        }
        return version

    def update_agent_with_versioning(
        self,
        agent_id: str,
        updates: Dict[str, Any],
        change_description: str = "",
        changed_by: str = "system"
    ) -> bool:
        """
        Update agent with automatic versioning

        This method:
        1. Retrieves current agent data
        2. Saves current state to versions array
        3. Applies updates
        4. Increments currentVersion
        5. Updates dateModified and lastActivity

        Args:
            agent_id: Agent ID (local or Azure)
            updates: Dict of fields to update
            change_description: Description of changes
            changed_by: Who made the change

        Returns:
            True if successful
        """
        # Get current agent data
        agent = self.db.get_agent(agent_id=agent_id)
        if not agent:
            agent = self.db.get_agent(azure_agent_id=agent_id)
        
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")

        # Initialize versions array if not exists
        if "versions" not in agent:
            agent["versions"] = []
        
        # Initialize currentVersion if not exists
        if "currentVersion" not in agent:
            agent["currentVersion"] = 1

        # Create version snapshot before update
        version_snapshot = self.create_version_snapshot(
            agent_data=agent,
            change_description=change_description,
            changed_by=changed_by
        )

        # Add snapshot to versions array
        agent["versions"].append(version_snapshot)

        # Apply updates
        now = datetime.utcnow().isoformat() + "Z"
        
        for key, value in updates.items():
            if key in ["name", "description", "instruction", "category", "status", "content"]:
                agent[key] = value
            elif key == "avatar_url":
                agent["avatarUrl"] = value
            elif key == "function_list":
                # Format function list
                function_list_formatted = []
                for func in value:
                    func_id = func.get("id") or func.get("function_id", "")
                    func_name = func.get("name") or func.get("function_name", "")
                    function_list_formatted.append(f"{func_id}<sep>{func_name}")
                agent["functionList"] = function_list_formatted
                agent["function"] = len(function_list_formatted) > 0
            elif key == "knowledge_base":
                # Format knowledge base
                kb_parts = []
                for kb in value:
                    kb_name = kb.get("name") or kb.get("kb_name", "")
                    kb_index = kb.get("index") or kb.get("kb_index", "")
                    kb_parts.extend([kb_name, kb_index])
                agent["knowledgeBase"] = "<sep>".join(kb_parts)
                agent["knowledge"] = len(kb_parts) > 0
            elif key == "sample_prompts":
                agent["samplePrompts"] = value
            elif key == "scenarios":
                agent["scenarios"] = value
            elif key == "maintainers":
                agent["maintainers"] = value

        # Increment version
        agent["currentVersion"] = agent["currentVersion"] + 1
        agent["dateModified"] = now
        agent["lastActivity"] = now

        # Update in database first to save version snapshot
        try:
            self.db.container.replace_item(item=agent["id"], body=agent)
            print(f"✅ Version {agent['currentVersion']} created in database")
        except Exception as e:
            print(f"❌ Error updating database with versioning: {e}")
            return False

        # Now update Azure agent using AgentManager's update_agent method
        if agent.get("azure_agent_id"):
            try:
                # Prepare updates for AgentManager.update_agent
                agent_updates = {}
                
                if "name" in updates:
                    agent_updates["name"] = updates["name"]
                if "instruction" in updates:
                    agent_updates["instructions"] = updates["instruction"]
                if "description" in updates:
                    agent_updates["description"] = updates["description"]
                if "category" in updates:
                    agent_updates["category"] = updates["category"]
                if "status" in updates:
                    agent_updates["status"] = updates["status"]
                if "avatar_url" in updates:
                    agent_updates["avatar_url"] = updates["avatar_url"]
                if "function_list" in updates:
                    agent_updates["function_list"] = updates["function_list"]
                if "knowledge_base" in updates:
                    agent_updates["knowledge_base"] = updates["knowledge_base"]
                if "sample_prompts" in updates:
                    agent_updates["sample_prompts"] = updates["sample_prompts"]
                if "scenarios" in updates:
                    agent_updates["scenarios"] = updates["scenarios"]
                if "maintainers" in updates:
                    agent_updates["maintainers"] = updates["maintainers"]
                
                # Use AgentManager's update_agent which handles both Azure and DB
                if agent_updates:
                    self.agent_manager.update_agent(
                        agent_id=agent["azure_agent_id"],
                        **agent_updates
                    )
            except Exception as e:
                print(f"⚠️  Warning: Could not update Azure agent: {e}")
                # Version is already saved in DB, so we can continue

        return True

    def get_version_history(
        self,
        agent_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get version history for an agent

        Args:
            agent_id: Agent ID
            limit: Maximum number of versions to return (most recent first)

        Returns:
            List of version snapshots
        """
        agent = self.db.get_agent(agent_id=agent_id)
        if not agent:
            agent = self.db.get_agent(azure_agent_id=agent_id)
        
        if not agent:
            return []

        versions = agent.get("versions", [])
        
        # Sort by version number descending
        versions_sorted = sorted(
            versions,
            key=lambda v: v.get("versionNumber", 0),
            reverse=True
        )

        if limit:
            return versions_sorted[:limit]
        
        return versions_sorted

    def get_version(
        self,
        agent_id: str,
        version_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific version snapshot

        Args:
            agent_id: Agent ID
            version_number: Version number to retrieve

        Returns:
            Version snapshot or None
        """
        versions = self.get_version_history(agent_id)
        
        for version in versions:
            if version.get("versionNumber") == version_number:
                return version
        
        return None

    def compare_versions(
        self,
        agent_id: str,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """
        Compare two versions and show differences

        Args:
            agent_id: Agent ID
            version1: First version number
            version2: Second version number

        Returns:
            Dict with comparison results
        """
        v1 = self.get_version(agent_id, version1)
        v2 = self.get_version(agent_id, version2)

        if not v1 or not v2:
            return {"error": "One or both versions not found"}

        snapshot1 = v1.get("snapshot", {})
        snapshot2 = v2.get("snapshot", {})

        differences = {}
        
        # Compare key fields
        for key in ["name", "description", "instruction", "category", "status"]:
            val1 = snapshot1.get(key)
            val2 = snapshot2.get(key)
            
            if val1 != val2:
                differences[key] = {
                    f"version_{version1}": val1,
                    f"version_{version2}": val2
                }

        return {
            "version1": version1,
            "version2": version2,
            "timestamp1": v1.get("timestamp"),
            "timestamp2": v2.get("timestamp"),
            "differences": differences
        }

    def rollback_to_version(
        self,
        agent_id: str,
        version_number: int,
        change_description: str = "",
        changed_by: str = "system"
    ) -> bool:
        """
        Rollback agent to a previous version

        Args:
            agent_id: Agent ID
            version_number: Version number to rollback to
            change_description: Description of rollback
            changed_by: Who performed the rollback

        Returns:
            True if successful
        """
        version = self.get_version(agent_id, version_number)
        
        if not version:
            raise ValueError(f"Version {version_number} not found")

        snapshot = version.get("snapshot", {})

        # Prepare updates from snapshot
        updates = {
            "name": snapshot.get("name"),
            "description": snapshot.get("description"),
            "instruction": snapshot.get("instruction"),
            "category": snapshot.get("category"),
            "status": snapshot.get("status"),
            "sample_prompts": snapshot.get("samplePrompts", []),
            "scenarios": snapshot.get("scenarios", []),
            "maintainers": snapshot.get("maintainers", [])
        }

        # Add avatar if exists
        if snapshot.get("avatarUrl"):
            updates["avatar_url"] = snapshot["avatarUrl"]

        # Parse and add function list
        if snapshot.get("functionList"):
            function_list = []
            for func_str in snapshot["functionList"]:
                parts = func_str.split("<sep>")
                if len(parts) == 2:
                    function_list.append({
                        "function_id": parts[0],
                        "function_name": parts[1]
                    })
            updates["function_list"] = function_list

        # Parse and add knowledge base
        if snapshot.get("knowledgeBase"):
            kb_str = snapshot["knowledgeBase"]
            kb_parts = kb_str.split("<sep>")
            knowledge_base = []
            for i in range(0, len(kb_parts), 2):
                if i + 1 < len(kb_parts):
                    knowledge_base.append({
                        "kb_name": kb_parts[i],
                        "kb_index": kb_parts[i + 1]
                    })
            updates["knowledge_base"] = knowledge_base

        # Update with versioning
        rollback_description = change_description or f"Rollback to version {version_number}"
        
        return self.update_agent_with_versioning(
            agent_id=agent_id,
            updates=updates,
            change_description=rollback_description,
            changed_by=changed_by
        )

    def get_current_version_number(self, agent_id: str) -> int:
        """
        Get current version number

        Args:
            agent_id: Agent ID

        Returns:
            Current version number
        """
        agent = self.db.get_agent(agent_id=agent_id)
        if not agent:
            agent = self.db.get_agent(azure_agent_id=agent_id)
        
        if not agent:
            return 0

        return agent.get("currentVersion", 1)
