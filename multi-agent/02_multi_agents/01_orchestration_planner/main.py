import json
import os
from typing import List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
from user_functions import user_functions
from azure.ai.projects.models import (
    FunctionTool,
    RequiredFunctionToolCall,
    SubmitToolOutputsAction,
    ToolOutput
)


class AgentOrchestrator:
    def __init__(
        self,
        api_key: str = None,
        endpoint: str = None,
        deployment_name: str = None,
        project_client=None
    ):
        if not all([api_key, endpoint, deployment_name]):
            load_dotenv()
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT')

        self.deployment_name = deployment_name
        self.aoai_client = AzureOpenAI(
            api_key=api_key,
            api_version="2023-05-15",
            azure_endpoint=endpoint
        )
        self.project_client = project_client
        self.messages = []
        self.agent_fleet = []
        self.functions = FunctionTool(functions=user_functions)

        # Initialize agent fleet
        self.load_agent_fleet()

    def load_agent_fleet(self):
        """Load agents belonging to specified fleet"""
        try:
            agent_list = self.project_client.agents.list_agents().data
            self.agent_fleet = [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "description": agent.description
                }
                for agent in agent_list
                if "group" in agent.metadata.keys()
                and agent.metadata["group"] == self.fleet_name
            ]
            print(
                f"Loaded {len(self.agent_fleet)} agents from fleet '{self.fleet_name}'")
        except Exception as e:
            print(f"Error loading agent fleet: {str(e)}")
            self.agent_fleet = []

    def get_agent_by_name(self, name: str) -> Optional[Dict]:
        """Get agent details by name"""
        return next(
            (agent for agent in self.agent_fleet if agent["name"] == name),
            None
        )

    def set_agent_fleet(self, agents: List[Dict]):
        self.agent_fleet = agents

    def initialize_conversation(self, system_message: str, user_message: str):
        self.messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    def get_assistant_content(self, messages_data: List[Dict]) -> List[str]:
        return [msg["content"] for msg in messages_data
                if msg.get("role") == "assistant"]

    def agent_execution(self, agent_id: str, task: str, context: str):
        thread = self.project_client.agents.create_thread()
        print(f"Created thread, ID: {thread.id}")

        message = self.project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=f'task: {task} \n\n context: {context}'
        )
        print(f"Created message, ID: {message.id}")

        run = self.project_client.agents.create_run(
            thread_id=thread.id,
            assistant_id=agent_id
        )
        print(f"Created run, ID: {run.id}")

        while run.status in ["queued", "in_progress", "requires_action"]:
            run = self.project_client.agents.get_run(
                thread_id=thread.id,
                run_id=run.id
            )

            if run.status == "requires_action" and isinstance(
                run.required_action,
                SubmitToolOutputsAction
            ):
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                if not tool_calls:
                    print("No tool calls provided - cancelling run")
                    self.project_client.agents.cancel_run(
                        thread_id=thread.id,
                        run_id=run.id
                    )
                    break

                tool_outputs = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, RequiredFunctionToolCall):
                        try:
                            print(f"Executing tool call: {tool_call}")
                            output = self.functions.execute(tool_call)
                            tool_outputs.append(
                                ToolOutput(
                                    tool_call_id=tool_call.id,
                                    output=output,
                                )
                            )
                        except Exception as e:
                            print(
                                f"Error executing tool_call {tool_call.id}: {e}")

                print(f"Tool outputs: {tool_outputs}")
                if tool_outputs:
                    self.project_client.agents.submit_tool_outputs_to_run(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )

            print(f"Current run status: {run.status}")

        print(f"Run completed with status: {run.status}")
        return self.project_client.agents.list_messages(thread_id=thread.id)

    def orchestrate(self) -> List[Dict]:
        response = self.aoai_client.chat.completions.create(
            model=self.deployment_name,
            messages=self.messages,
            temperature=0.7,
            max_tokens=1000
        )

        llm_response = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": llm_response})

        while "[__AGENT__]" in llm_response:
            try:
                agent_name = llm_response.split("[__AGENT__]")[1].split(";")[0]
                agent_task = llm_response.split("[__TASK__]")[1]

                agent_id = next(
                    (agent["id"] for agent in self.agent_fleet
                     if agent["name"] == agent_name),
                    None
                )

                if agent_id:
                    agent_messages = self.agent_execution(
                        agent_id,
                        agent_task,
                        json.dumps(self.messages[1:])
                    )

                    agent_contents = self.get_assistant_content(
                        agent_messages.data)
                    if agent_contents:
                        for agent_content in agent_contents:
                            self.messages.append({
                                "role": "assistant",
                                "content": f"[__AGENT__({agent_name})]{agent_content}"
                            })

                    response = self.aoai_client.chat.completions.create(
                        model=self.deployment_name,
                        messages=self.messages,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    llm_response = response.choices[0].message.content
                    self.messages.append(
                        {"role": "assistant", "content": llm_response}
                    )
                else:
                    print(f"Agent {agent_name} not found in fleet")
                    break
            except Exception as e:
                print(f"Error in orchestration: {str(e)}")
                break

        return self.messages


if __name__ == "__main__":
    orchestrator = AgentOrchestrator()
    agent_fleet = [
        {"id": "agent1", "name": "SearchAgent"},
        {"id": "agent2", "name": "CalculatorAgent"}
    ]
    orchestrator.set_agent_fleet(agent_fleet)
    orchestrator.initialize_conversation(
        "You are a helpful assistant that can use multiple agents.",
        "Can you help me calculate 2+2?"
    )
    messages = orchestrator.orchestrate()
    for msg in messages:
        print(f"{msg['role']}: {msg['content']}\n")
