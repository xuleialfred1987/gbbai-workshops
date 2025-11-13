"""
Azure AI Foundry Agent Utilities

This module provides helper functions for managing Azure AI agents, threads,
and executing agent runs with function calling support.
"""

import json
import time
import uuid
from typing import Dict, List, Callable, Optional, Any
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import FunctionTool, ToolSet
from agent_db import AgentDB


class AgentManager:
    """Manager class for Azure AI Foundry Agents with Azure Cosmos DB metadata storage"""

    def __init__(self, project_client: AIProjectClient):
        """
        Initialize the AgentManager

        Args:
            project_client: AIProjectClient instance
        
        Note:
            Cosmos DB connection is configured via AZURE_COSMOS_ENDPOINT environment variable.
            Authentication uses DefaultAzureCredential (AAD).
        """
        self.client = project_client
        self.db = AgentDB()

    # ============================================================================
    # Agent Operations
    # ============================================================================

    def create_agent(
        self,
        model: str,
        name: str,
        instructions: str,
        functions: Optional[Dict[str, Callable]] = None,
        tools: Optional[Any] = None,  # For Bing, Azure AI Search, etc.
        temperature: float = 1.0,
        top_p: float = 1.0,
        # Extended metadata for local DB
        local_id: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        status: str = "draft",
        avatar_url: Optional[str] = None,
        function_list: Optional[List[Dict[str, str]]] = None,
        knowledge_base: Optional[List[Dict[str, str]]] = None,
        sample_prompts: Optional[List[str]] = None,
        scenarios: Optional[List[Dict[str, str]]] = None,
        maintainers: Optional[List[Dict[str, str]]] = None
    ) -> Any:
        """
        Create a new agent with optional function calling and store metadata in local DB

        Args:
            model: Model deployment name
            name: Agent name
            instructions: System instructions for the agent
            functions: Optional dict of function name -> callable
            tools: Optional tool definitions (e.g., BingGroundingTool.definitions)
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            local_id: Local unique ID (auto-generated if not provided)
            description: Agent description
            category: Agent category (e.g., "Productivity", "HR")
            status: Status (draft, published, archived)
            avatar_url: Avatar image URL
            function_list: List of function definitions for metadata
            knowledge_base: List of knowledge base definitions
            sample_prompts: List of sample prompts
            scenarios: List of scenario tags
            maintainers: List of maintainer info

        Returns:
            Agent object
        """
        agent_tools = None

        # Build toolset from both custom functions and pre-defined tools
        if functions or tools:
            # If we have ONLY pre-defined tools (no functions), pass them directly
            if tools and not functions:
                # Tools can be a list or have a .definitions attribute
                if isinstance(tools, list):
                    agent_tools = tools
                elif hasattr(tools, 'definitions'):
                    agent_tools = tools.definitions
                else:
                    agent_tools = [tools]
            else:
                # We have functions (with or without tools), use ToolSet
                toolset = ToolSet()
                
                # Add custom Python function tools
                if functions:
                    function_tool = FunctionTool(functions=set(functions.values()))
                    toolset.add(function_tool)
                
                # Add pre-defined tools (Bing, Azure AI Search, etc.)
                if tools:
                    if isinstance(tools, list):
                        # Append each tool definition individually
                        for tool_def in tools:
                            toolset.definitions.append(tool_def)
                    elif hasattr(tools, 'definitions'):
                        # Append each definition from the tool's definitions list
                        for tool_def in tools.definitions:
                            toolset.definitions.append(tool_def)
                
                agent_tools = toolset.definitions

        # Create agent in Azure
        agent = self.client.agents.create_agent(
            model=model,
            name=name,
            instructions=instructions,
            tools=agent_tools,
            temperature=temperature,
            top_p=top_p
        )

        print(f"✅ Agent created: {agent.id}")
        print(f"📝 Name: {name}")
        print(f"🤖 Model: {model}")
        if functions:
            print(
                f"🛠️  Functions: {len(functions)} ({', '.join(functions.keys())})")

        # Store metadata in local DB
        try:
            if not local_id:
                local_id = str(uuid.uuid4())
            
            # Auto-generate function_list if functions provided but function_list not set
            if functions and not function_list:
                function_list = []
                for func_name in functions.keys():
                    func_id = str(uuid.uuid4())
                    function_list.append({
                        "id": func_id,
                        "name": func_name
                    })
            
            # Ensure function_list is in proper format (list of dicts)
            formatted_function_list = function_list if function_list else []
            
            self.db.create_agent(
                agent_id=local_id,
                azure_agent_id=agent.id,
                name=name,
                instruction=instructions,
                description=description,
                category=category,
                status=status,
                avatar_url=avatar_url,
                function=bool(functions),
                knowledge=bool(knowledge_base),
                function_list=formatted_function_list,
                knowledge_base=knowledge_base,
                sample_prompts=sample_prompts,
                scenarios=scenarios,
                maintainers=maintainers
            )
            print(f"💾 Metadata saved to local DB: {local_id}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save metadata to DB: {e}")

        return agent

    def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        functions: Optional[Dict[str, Callable]] = None,
        # Extended metadata updates
        description: Optional[str] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
        avatar_url: Optional[str] = None,
        function_list: Optional[List[Dict[str, str]]] = None,
        knowledge_base: Optional[List[Dict[str, str]]] = None,
        sample_prompts: Optional[List[str]] = None,
        scenarios: Optional[List[Dict[str, str]]] = None,
        maintainers: Optional[List[Dict[str, str]]] = None
    ) -> Any:
        """
        Update an existing agent in Azure and local DB

        Args:
            agent_id: Azure agent ID to update
            name: New agent name
            instructions: New instructions
            model: New model deployment
            functions: New functions dict
            description: New description
            category: New category
            status: New status
            avatar_url: New avatar URL
            function_list: New function list
            knowledge_base: New knowledge base list
            sample_prompts: New sample prompts
            scenarios: New scenarios
            maintainers: New maintainers

        Returns:
            Updated agent object
        """
        update_params = {}

        if name is not None:
            update_params['name'] = name
        if instructions is not None:
            update_params['instructions'] = instructions
        if model is not None:
            update_params['model'] = model
        if functions is not None:
            function_tool = FunctionTool(functions=set(functions.values()))
            toolset = ToolSet()
            toolset.add(function_tool)
            update_params['toolset'] = toolset

        # Update in Azure
        agent = self.client.agents.update_agent(
            agent_id=agent_id,
            **update_params
        )

        print(f"✅ Agent updated: {agent_id}")

        # Update metadata in local DB
        try:
            db_updates = {}
            if name is not None:
                db_updates['name'] = name
            if instructions is not None:
                db_updates['instruction'] = instructions
            if model is not None:
                db_updates['model'] = model
            if description is not None:
                db_updates['description'] = description
            if category is not None:
                db_updates['category'] = category
            if status is not None:
                db_updates['status'] = status
            if avatar_url is not None:
                db_updates['avatar_url'] = avatar_url
            if function_list is not None:
                db_updates['function_list'] = function_list
            if knowledge_base is not None:
                db_updates['knowledge_base'] = knowledge_base
            if sample_prompts is not None:
                db_updates['sample_prompts'] = sample_prompts
            if scenarios is not None:
                db_updates['scenarios'] = scenarios
            if maintainers is not None:
                db_updates['maintainers'] = maintainers
            
            if db_updates:
                self.db.update_agent(azure_agent_id=agent_id, **db_updates)
                print(f"💾 Metadata updated in local DB")
        except Exception as e:
            print(f"⚠️  Warning: Could not update metadata in DB: {e}")

        return agent

    def get_agent(self, agent_id: str) -> Any:
        """
        Get agent by ID

        Args:
            agent_id: Agent ID

        Returns:
            Agent object
        """
        return self.client.agents.get(agent_id)

    def list_agents(self, limit: int = 20) -> List[Any]:
        """
        List all agents

        Args:
            limit: Maximum number of agents to return

        Returns:
            List of agent objects
        """
        agents = self.client.agents.list_agents(limit=limit)
        return list(agents)

    def delete_agent(self, agent_id: str, silent: bool = False) -> bool:
        """
        Delete an agent from Azure and local DB

        Args:
            agent_id: Azure agent ID to delete
            silent: If True, suppress output messages

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from Azure
            self.client.agents.delete(agent_id)
            if not silent:
                print(f"✅ Agent deleted from Azure: {agent_id}")
            
            # Delete from local DB
            try:
                self.db.delete_agent(azure_agent_id=agent_id)
                if not silent:
                    print(f"💾 Agent metadata deleted from local DB")
            except Exception as e:
                if not silent:
                    print(f"⚠️  Warning: Could not delete metadata from DB: {e}")
            
            return True
        except Exception as e:
            if not silent:
                print(f"⚠️  Could not delete agent {agent_id}: {e}")
            return False

    # ============================================================================
    # Local DB Query Operations
    # ============================================================================

    def get_agent_metadata(self, agent_id: Optional[str] = None, azure_agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get agent metadata from local DB

        Args:
            agent_id: Local agent ID
            azure_agent_id: Azure agent ID

        Returns:
            Agent metadata dict
        """
        try:
            return self.db.get_agent(agent_id=agent_id, azure_agent_id=azure_agent_id)
        except Exception as e:
            print(f"⚠️  Error retrieving agent metadata: {e}")
            return None

    def list_agents_metadata(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List agents from local DB with filtering

        Args:
            status: Filter by status
            category: Filter by category
            limit: Maximum number of results

        Returns:
            List of agent metadata dicts
        """
        try:
            return self.db.list_agents(status=status, category=category, limit=limit)
        except Exception as e:
            print(f"⚠️  Error listing agents: {e}")
            return []

    def search_agents_metadata(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search agents in local DB

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching agents
        """
        try:
            return self.db.search_agents(query=query, limit=limit)
        except Exception as e:
            print(f"⚠️  Error searching agents: {e}")
            return []

    def get_db_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dict with stats
        """
        try:
            return self.db.get_stats()
        except Exception as e:
            print(f"⚠️  Error getting DB stats: {e}")
            return {}

    def export_agent_metadata(self, agent_id: str, output_path: str):
        """
        Export agent metadata to JSON file

        Args:
            agent_id: Local agent ID
            output_path: Output file path
        """
        try:
            self.db.export_agent(agent_id=agent_id, output_path=output_path)
            print(f"✅ Agent metadata exported to {output_path}")
        except Exception as e:
            print(f"⚠️  Error exporting agent: {e}")

    def import_agent_metadata(self, input_path: str) -> Optional[str]:
        """
        Import agent metadata from JSON file (creates DB record only, not Azure agent)

        Args:
            input_path: Input file path

        Returns:
            Created agent_id or None
        """
        try:
            agent_id = self.db.import_agent(input_path=input_path)
            print(f"✅ Agent metadata imported: {agent_id}")
            return agent_id
        except Exception as e:
            print(f"⚠️  Error importing agent: {e}")
            return None

    # ============================================================================
    # Thread Operations
    # ============================================================================

    def create_thread(self, metadata: Optional[Dict[str, str]] = None) -> Any:
        """
        Create a new conversation thread

        Args:
            metadata: Optional metadata dict

        Returns:
            Thread object
        """
        thread = self.client.agents.threads.create(metadata=metadata)
        print(f"✅ Thread created: {thread.id}")
        return thread

    def delete_thread(self, thread_id: str, silent: bool = False) -> bool:
        """
        Delete a thread

        Args:
            thread_id: Thread ID to delete
            silent: If True, suppress output messages

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.agents.threads.delete(thread_id)
            if not silent:
                print(f"✅ Thread deleted: {thread_id}")
            return True
        except Exception as e:
            if not silent:
                print(f"⚠️  Could not delete thread {thread_id}: {e}")
            return False

    def add_message(
        self,
        thread_id: str,
        content: str,
        role: str = "user"
    ) -> Any:
        """
        Add a message to a thread

        Args:
            thread_id: Thread ID
            content: Message content
            role: Message role (user/assistant)

        Returns:
            Message object
        """
        message = self.client.agents.messages.create(
            thread_id=thread_id,
            role=role,
            content=content
        )
        return message

    def get_messages(self, thread_id: str, limit: int = 50) -> List[Any]:
        """
        Get messages from a thread

        Args:
            thread_id: Thread ID
            limit: Maximum number of messages to return

        Returns:
            List of message objects
        """
        messages = self.client.agents.messages.list(
            thread_id=thread_id,
            limit=limit
        )
        return list(messages)

    # ============================================================================
    # Run Operations
    # ============================================================================

    def run_agent(
        self,
        thread_id: str,
        agent_id: str,
        user_message: str,
        functions: Optional[Dict[str, Callable]] = None,
        max_iterations: int = 10,
        verbose: bool = True,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        max_prompt_tokens: Optional[int] = None
    ) -> str:
        """
        Run an agent on a thread with automatic function calling

        Args:
            thread_id: Thread ID
            agent_id: Agent ID
            user_message: User's message
            functions: Dict of function name -> callable for function calling
            max_iterations: Max iterations for function calling loop
            verbose: If True, print progress messages
            temperature: Sampling temperature (0-2). Higher = more random
            top_p: Nucleus sampling parameter (0-1)
            max_completion_tokens: Maximum tokens in the completion
            max_prompt_tokens: Maximum tokens in the prompt

        Returns:
            Assistant's response text
        """
        # Add user message
        self.add_message(thread_id, user_message)

        if verbose:
            print(f"📨 Message sent: {user_message[:50]}...")

        # Prepare optional parameters
        run_params = {
            "thread_id": thread_id,
            "agent_id": agent_id
        }

        if temperature is not None:
            run_params["temperature"] = temperature
        if top_p is not None:
            run_params["top_p"] = top_p
        if max_completion_tokens is not None:
            run_params["max_completion_tokens"] = max_completion_tokens
        if max_prompt_tokens is not None:
            run_params["max_prompt_tokens"] = max_prompt_tokens

        # Create run
        run = self.client.agents.runs.create(**run_params)

        if verbose:
            print("🔄 Running agent...")

        # Process run with function calling support
        iteration = 0
        while run.status in ["queued", "in_progress", "requires_action"]:
            iteration += 1
            if iteration > max_iterations:
                print(f"⚠️  Max iterations ({max_iterations}) reached")
                break

            # Wait briefly before checking status
            time.sleep(0.5)

            # Get latest run status
            run = self.client.agents.runs.get(
                thread_id=thread_id,
                run_id=run.id
            )

            # Handle function calls
            if run.status == "requires_action" and functions:
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                if verbose:
                    print(f"\n🔧 Function calls detected: {len(tool_calls)}")

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if verbose:
                        print(f"📞 Calling: {function_name}({function_args})")

                    # Execute the function
                    if function_name in functions:
                        try:
                            result = functions[function_name](**function_args)
                            if verbose:
                                print(f"📊 Result: {result}")

                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": result
                            })
                        except Exception as e:
                            print(f"❌ Error calling {function_name}: {e}")
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": json.dumps({"error": str(e)})
                            })
                    else:
                        print(
                            f"⚠️  Function {function_name} not found in provided functions")

                # Submit function results
                if tool_outputs:
                    self.client.agents.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )

        if verbose:
            print(f"✅ Run completed: {run.status}")

        # Get the assistant's response
        messages = self.get_messages(thread_id)

        for msg in messages:
            if msg.role == "assistant":
                if msg.text_messages:
                    response = "\n".join([
                        text_msg.text.value
                        for text_msg in msg.text_messages
                    ])
                    return response

        return ""

    def run_agent_simple(
        self,
        thread_id: str,
        agent_id: str,
        user_message: str,
        verbose: bool = True,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        max_prompt_tokens: Optional[int] = None
    ) -> str:
        """
        Simple run without function calling (uses create_and_process)

        Args:
            thread_id: Thread ID
            agent_id: Agent ID
            user_message: User's message
            verbose: If True, print progress messages
            temperature: Sampling temperature (0-2). Higher = more random
            top_p: Nucleus sampling parameter (0-1)
            max_completion_tokens: Maximum tokens in the completion
            max_prompt_tokens: Maximum tokens in the prompt

        Returns:
            Assistant's response text
        """
        # Add user message
        self.add_message(thread_id, user_message)

        if verbose:
            print(f"📨 Message sent: {user_message[:50]}...")

        # Prepare optional parameters
        run_params = {
            "thread_id": thread_id,
            "agent_id": agent_id
        }

        if temperature is not None:
            run_params["temperature"] = temperature
        if top_p is not None:
            run_params["top_p"] = top_p
        if max_completion_tokens is not None:
            run_params["max_completion_tokens"] = max_completion_tokens
        if max_prompt_tokens is not None:
            run_params["max_prompt_tokens"] = max_prompt_tokens

        # Run agent (auto-processes to completion)
        run = self.client.agents.runs.create_and_process(**run_params)

        if verbose:
            print(f"✅ Run completed: {run.status}")

        # Get response
        messages = self.get_messages(thread_id)

        for msg in messages:
            if msg.role == "assistant":
                if msg.text_messages:
                    response = "\n".join([
                        text_msg.text.value
                        for text_msg in msg.text_messages
                    ])
                    return response

        return ""

    def stream_agent(
        self,
        thread_id: str,
        agent_id: str,
        user_message: str,
        callback: Optional[Callable[[str], None]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        max_prompt_tokens: Optional[int] = None
    ) -> str:
        """
        Run agent with streaming responses

        Args:
            thread_id: Thread ID
            agent_id: Agent ID
            user_message: User's message
            callback: Optional callback function for each text chunk
            temperature: Sampling temperature (0-2). Higher = more random
            top_p: Nucleus sampling parameter (0-1)
            max_completion_tokens: Maximum tokens in the completion
            max_prompt_tokens: Maximum tokens in the prompt

        Returns:
            Complete response text
        """
        # Add user message
        self.add_message(thread_id, user_message)

        response_text = []

        # Prepare optional parameters
        stream_params = {
            "thread_id": thread_id,
            "agent_id": agent_id
        }

        if temperature is not None:
            stream_params["temperature"] = temperature
        if top_p is not None:
            stream_params["top_p"] = top_p
        if max_completion_tokens is not None:
            stream_params["max_completion_tokens"] = max_completion_tokens
        if max_prompt_tokens is not None:
            stream_params["max_prompt_tokens"] = max_prompt_tokens

        # Create and stream the run
        with self.client.agents.runs.stream(**stream_params) as stream:
            for event_type, event_data, _ in stream:
                # Handle text delta events for streaming
                if event_type == "thread.message.delta":
                    if hasattr(event_data, "delta") and hasattr(event_data.delta, "content"):
                        for content in event_data.delta.content:
                            if hasattr(content, "text") and hasattr(content.text, "value"):
                                text_chunk = content.text.value
                                response_text.append(text_chunk)

                                if callback:
                                    callback(text_chunk)
                                else:
                                    print(text_chunk, end='', flush=True)

        if not callback:
            print()  # New line after streaming

        return "".join(response_text)

    # ============================================================================
    # Cleanup Operations
    # ============================================================================

    def cleanup(
        self,
        agent_ids: Optional[List[str]] = None,
        thread_ids: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, int]:
        """
        Cleanup agents and threads

        Args:
            agent_ids: List of agent IDs to delete (None = skip)
            thread_ids: List of thread IDs to delete (None = skip)
            verbose: If True, print progress messages

        Returns:
            Dict with counts of deleted agents and threads
        """
        deleted = {"agents": 0, "threads": 0}

        # Delete threads
        if thread_ids:
            for thread_id in thread_ids:
                if self.delete_thread(thread_id, silent=not verbose):
                    deleted["threads"] += 1

        # Delete agents
        if agent_ids:
            for agent_id in agent_ids:
                if self.delete_agent(agent_id, silent=not verbose):
                    deleted["agents"] += 1

        if verbose:
            print(f"\n🧹 Cleanup completed:")
            print(f"   - Agents deleted: {deleted['agents']}")
            print(f"   - Threads deleted: {deleted['threads']}")

        return deleted


# ============================================================================
# Standalone Helper Functions
# ============================================================================

def create_agent_manager(project_client: AIProjectClient) -> AgentManager:
    """
    Factory function to create an AgentManager

    Args:
        project_client: AIProjectClient instance

    Returns:
        AgentManager instance
    """
    return AgentManager(project_client)


def format_messages(messages: List[Any]) -> str:
    """
    Format messages for display

    Args:
        messages: List of message objects

    Returns:
        Formatted string
    """
    output = []

    for msg in messages:
        role = msg.role.upper()
        output.append(f"\n{'=' * 80}")
        output.append(f"{role}:")
        output.append(f"{'=' * 80}")

        if msg.text_messages:
            for text_msg in msg.text_messages:
                output.append(text_msg.text.value)

    output.append(f"{'=' * 80}\n")

    return "\n".join(output)
