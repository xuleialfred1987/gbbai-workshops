"""
Agent Metadata Database Manager

This module provides Azure Cosmos DB management for storing agent metadata
alongside Azure AI Foundry Agent Service.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.identity import DefaultAzureCredential


class AgentDB:
    """Azure Cosmos DB manager for agent metadata"""

    def __init__(
        self,
        database_name: str = "Acsp",
        container_name: str = "Agents"
    ):
        """
        Initialize the AgentDB with Cosmos DB using AAD authentication

        Args:
            database_name: Database name
            container_name: Container name
        """
        # Get endpoint from environment variable
        self.endpoint = os.environ.get("AZURE_COSMOS_ENDPOINT")
        if not self.endpoint:
            raise ValueError("AZURE_COSMOS_ENDPOINT environment variable is required")
        
        self.database_name = database_name
        self.container_name = container_name
        
        # Initialize Cosmos client with AAD authentication
        aad_credentials = DefaultAzureCredential()
        self.client = CosmosClient(self.endpoint, aad_credentials)
        
        self.database = None
        self.container = None
        self._init_db()

    def _init_db(self):
        """Initialize database and container"""
        try:
            # Create database if it doesn't exist
            self.database = self.client.create_database_if_not_exists(id=self.database_name)
            
            # Create container if it doesn't exist
            # Partition key: /id (each agent is its own partition)
            self.container = self.database.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path="/id"),
                offer_throughput=400  # Adjust as needed
            )
            
            print(f"✅ Cosmos DB initialized: {self.database_name}/{self.container_name}")
        except exceptions.CosmosHttpResponseError as e:
            print(f"❌ Error initializing Cosmos DB: {e.message}")
            raise

    def close(self):
        """Close database connection (no-op for Cosmos DB SDK)"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ============================================================================
    # Agent CRUD Operations
    # ============================================================================

    def create_agent(
        self,
        agent_id: str,
        azure_agent_id: str,
        name: str,
        instruction: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        status: str = "draft",
        avatar_url: Optional[str] = None,
        function: bool = False,
        knowledge: bool = False,
        assistant: bool = False,
        assistant_name: Optional[str] = None,
        function_list: Optional[List[Dict[str, str]]] = None,
        knowledge_base: Optional[List[Dict[str, str]]] = None,
        sample_prompts: Optional[List[str]] = None,
        scenarios: Optional[List[Dict[str, str]]] = None,
        maintainers: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Create a new agent record in Cosmos DB

        Args:
            agent_id: Local unique ID
            azure_agent_id: Azure Agent Service ID
            name: Agent name
            instruction: System instructions
            description: Agent description
            category: Agent category
            status: Status (draft, published, archived)
            avatar_url: Avatar image URL
            function: Whether agent has functions
            knowledge: Whether agent has knowledge base
            assistant: Whether this is an assistant
            assistant_name: Assistant name
            function_list: List of function definitions
            knowledge_base: List of knowledge base definitions
            sample_prompts: List of sample prompts
            scenarios: List of scenario tags
            maintainers: List of maintainer info
            **kwargs: Additional fields

        Returns:
            agent_id
        """
        now = datetime.utcnow().isoformat() + "Z"
        
        # Format function list to Cosmos DB format
        function_list_formatted = []
        if function_list:
            for func in function_list:
                func_id = func.get("id") or func.get("function_id", "")
                func_name = func.get("name") or func.get("function_name", "")
                function_list_formatted.append(f"{func_id}<sep>{func_name}")
        
        # Format knowledge base to Cosmos DB format
        knowledge_base_formatted = ""
        if knowledge_base:
            kb_parts = []
            for kb in knowledge_base:
                kb_name = kb.get("name") or kb.get("kb_name", "")
                kb_index = kb.get("index") or kb.get("kb_index", "")
                kb_parts.extend([kb_name, kb_index])
            knowledge_base_formatted = "<sep>".join(kb_parts)
        
        # Create document
        document = {
            "id": agent_id,
            "azure_agent_id": azure_agent_id,
            "name": name,
            "description": description,
            "instruction": instruction,
            "category": category,
            "status": status,
            "avatarUrl": avatar_url,
            "function": function,
            "knowledge": knowledge,
            "assistant": assistant,
            "assistantName": assistant_name,
            "functionList": function_list_formatted,
            "knowledgeBase": knowledge_base_formatted,
            "samplePrompts": sample_prompts or [],
            "scenarios": scenarios or [],
            "maintainers": maintainers or [],
            "dateCreated": now,
            "dateModified": now,
            "lastActivity": now,
            "content": kwargs.get("content", ""),
            "_attachments": "attachments/"
        }
        
        try:
            self.container.create_item(body=document)
            return agent_id
        except exceptions.CosmosResourceExistsError:
            raise ValueError(f"Agent with id {agent_id} already exists")
        except exceptions.CosmosHttpResponseError as e:
            raise Exception(f"Error creating agent: {e.message}")

    def get_agent(self, agent_id: Optional[str] = None, azure_agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get agent by local ID or Azure agent ID

        Args:
            agent_id: Local agent ID
            azure_agent_id: Azure agent ID

        Returns:
            Agent dict with all related data
        """
        if not agent_id and not azure_agent_id:
            raise ValueError("Either agent_id or azure_agent_id must be provided")
        
        try:
            if agent_id:
                # Direct read by ID (most efficient)
                return self.container.read_item(item=agent_id, partition_key=agent_id)
            else:
                # Query by azure_agent_id
                query = "SELECT * FROM c WHERE c.azure_agent_id = @azure_agent_id"
                parameters = [{"name": "@azure_agent_id", "value": azure_agent_id}]
                
                items = list(self.container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True
                ))
                
                return items[0] if items else None
                
        except exceptions.CosmosResourceNotFoundError:
            return None
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error retrieving agent: {e.message}")
            return None

    def update_agent(
        self,
        agent_id: Optional[str] = None,
        azure_agent_id: Optional[str] = None,
        **updates
    ) -> bool:
        """
        Update agent record

        Args:
            agent_id: Local agent ID
            azure_agent_id: Azure agent ID
            **updates: Fields to update

        Returns:
            True if successful
        """
        if not agent_id and not azure_agent_id:
            raise ValueError("Either agent_id or azure_agent_id must be provided")
        
        # Get current agent
        agent = self.get_agent(agent_id=agent_id, azure_agent_id=azure_agent_id)
        if not agent:
            return False
        
        now = datetime.utcnow().isoformat() + "Z"
        
        # Update fields
        if "name" in updates:
            agent["name"] = updates["name"]
        if "description" in updates:
            agent["description"] = updates["description"]
        if "instruction" in updates:
            agent["instruction"] = updates["instruction"]
        if "category" in updates:
            agent["category"] = updates["category"]
        if "status" in updates:
            agent["status"] = updates["status"]
        if "avatar_url" in updates:
            agent["avatarUrl"] = updates["avatar_url"]
        if "avatarUrl" in updates:
            agent["avatarUrl"] = updates["avatarUrl"]
        if "function" in updates:
            agent["function"] = updates["function"]
        if "knowledge" in updates:
            agent["knowledge"] = updates["knowledge"]
        if "assistant" in updates:
            agent["assistant"] = updates["assistant"]
        if "assistant_name" in updates:
            agent["assistantName"] = updates["assistant_name"]
        if "assistantName" in updates:
            agent["assistantName"] = updates["assistantName"]
        if "content" in updates:
            agent["content"] = updates["content"]
        
        # Update function list
        if "function_list" in updates:
            function_list_formatted = []
            for func in updates["function_list"]:
                func_id = func.get("id") or func.get("function_id", "")
                func_name = func.get("name") or func.get("function_name", "")
                function_list_formatted.append(f"{func_id}<sep>{func_name}")
            agent["functionList"] = function_list_formatted
        
        # Update knowledge base
        if "knowledge_base" in updates:
            kb_parts = []
            for kb in updates["knowledge_base"]:
                kb_name = kb.get("name") or kb.get("kb_name", "")
                kb_index = kb.get("index") or kb.get("kb_index", "")
                kb_parts.extend([kb_name, kb_index])
            agent["knowledgeBase"] = "<sep>".join(kb_parts)
        
        # Update sample prompts
        if "sample_prompts" in updates:
            agent["samplePrompts"] = updates["sample_prompts"]
        
        # Update scenarios
        if "scenarios" in updates:
            agent["scenarios"] = updates["scenarios"]
        
        # Update maintainers
        if "maintainers" in updates:
            agent["maintainers"] = updates["maintainers"]
        
        # Update lastReleaseValidation (deployment/monitoring metadata)
        if "lastReleaseValidation" in updates:
            agent["lastReleaseValidation"] = updates["lastReleaseValidation"]
        
        # Update governanceFeedback (governance/stakeholder feedback)
        if "governanceFeedback" in updates:
            agent["governanceFeedback"] = updates["governanceFeedback"]
        
        # Update timestamps
        agent["dateModified"] = now
        agent["lastActivity"] = now
        
        try:
            self.container.replace_item(item=agent["id"], body=agent)
            return True
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error updating agent: {e.message}")
            return False

    def delete_agent(self, agent_id: Optional[str] = None, azure_agent_id: Optional[str] = None) -> bool:
        """
        Delete agent record

        Args:
            agent_id: Local agent ID
            azure_agent_id: Azure agent ID

        Returns:
            True if successful
        """
        if not agent_id and not azure_agent_id:
            raise ValueError("Either agent_id or azure_agent_id must be provided")
        
        # Get agent to find its ID
        agent = self.get_agent(agent_id=agent_id, azure_agent_id=azure_agent_id)
        if not agent:
            return False
        
        try:
            self.container.delete_item(item=agent["id"], partition_key=agent["id"])
            return True
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error deleting agent: {e.message}")
            return False

    def list_agents(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List agents with optional filtering

        Args:
            status: Filter by status
            category: Filter by category
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of agent dicts
        """
        # Build query
        query = "SELECT * FROM c WHERE 1=1"
        parameters = []
        
        if status:
            query += " AND c.status = @status"
            parameters.append({"name": "@status", "value": status})
        
        if category:
            query += " AND c.category = @category"
            parameters.append({"name": "@category", "value": category})
        
        query += " ORDER BY c.dateModified DESC"
        
        if limit:
            query += f" OFFSET {offset} LIMIT {limit}"
        
        try:
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return items
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error listing agents: {e.message}")
            return []

    def search_agents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search agents by name, description, or instruction

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching agents
        """
        # Cosmos DB SQL query with CONTAINS for text search
        sql_query = """
            SELECT * FROM c 
            WHERE CONTAINS(c.name, @query) 
               OR CONTAINS(c.description, @query) 
               OR CONTAINS(c.instruction, @query)
            ORDER BY c.dateModified DESC
            OFFSET 0 LIMIT @limit
        """
        
        parameters = [
            {"name": "@query", "value": query},
            {"name": "@limit", "value": limit}
        ]
        
        try:
            items = list(self.container.query_items(
                query=sql_query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return items
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error searching agents: {e.message}")
            return []

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dict with stats
        """
        try:
            # Fetch all agents and aggregate in Python (avoids GROUP BY issues)
            query = "SELECT c.status, c.category FROM c"
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            # Calculate statistics
            total = len(items)
            by_status = {}
            by_category = {}
            
            for item in items:
                # Count by status
                status = item.get("status", "unknown")
                by_status[status] = by_status.get(status, 0) + 1
                
                # Count by category
                category = item.get("category")
                if category:
                    by_category[category] = by_category.get(category, 0) + 1
            
            return {
                "total_agents": total,
                "by_status": by_status,
                "by_category": by_category
            }
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error getting stats: {e.message}")
            return {"total_agents": 0, "by_status": {}, "by_category": {}}

    def export_agent(self, agent_id: str, output_path: str):
        """
        Export agent metadata to JSON file

        Args:
            agent_id: Agent ID to export
            output_path: Output file path
        """
        agent = self.get_agent(agent_id=agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(agent, f, indent=2, ensure_ascii=False)

    def import_agent(self, input_path: str) -> str:
        """
        Import agent metadata from JSON file

        Args:
            input_path: Input file path

        Returns:
            Created agent_id
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure required fields
        if "id" not in data:
            data["id"] = str(uuid.uuid4())
        if "azure_agent_id" not in data:
            raise ValueError("azure_agent_id is required")
        
        # Parse function list if needed
        function_list = []
        if "functionList" in data and isinstance(data["functionList"], list):
            for item in data["functionList"]:
                if isinstance(item, str) and "<sep>" in item:
                    parts = item.split("<sep>")
                    function_list.append({
                        "function_id": parts[0],
                        "function_name": parts[1] if len(parts) > 1 else ""
                    })
                elif isinstance(item, dict):
                    function_list.append(item)
        
        # Parse knowledge base if needed
        knowledge_base = []
        if "knowledgeBase" in data and isinstance(data["knowledgeBase"], str):
            kb_parts = data["knowledgeBase"].split("<sep>")
            for i in range(0, len(kb_parts), 2):
                if i + 1 < len(kb_parts):
                    knowledge_base.append({
                        "kb_name": kb_parts[i],
                        "kb_index": kb_parts[i + 1]
                    })
        
        # Map fields
        return self.create_agent(
            agent_id=data["id"],
            azure_agent_id=data["azure_agent_id"],
            name=data.get("name", ""),
            instruction=data.get("instruction", ""),
            description=data.get("description"),
            category=data.get("category"),
            status=data.get("status", "draft"),
            avatar_url=data.get("avatarUrl"),
            function=data.get("function", False),
            knowledge=data.get("knowledge", False),
            assistant=data.get("assistant", False),
            assistant_name=data.get("assistantName"),
            function_list=function_list if function_list else None,
            knowledge_base=knowledge_base if knowledge_base else None,
            sample_prompts=data.get("samplePrompts"),
            scenarios=data.get("scenarios"),
            maintainers=data.get("maintainers"),
            content=data.get("content")
        )
