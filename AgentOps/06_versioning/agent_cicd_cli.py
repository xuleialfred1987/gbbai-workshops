"""
Agent CI/CD Command Line Tool

This script provides a command-line interface for the agent CI/CD pipeline.
It can be used in automated workflows, GitHub Actions, or Azure DevOps pipelines.

Usage:
    python agent_cicd_cli.py update --agent-id <id> --config config.json
    python agent_cicd_cli.py evaluate --agent-id <id> --test-queries queries.json
    python agent_cicd_cli.py rollback --agent-id <id> --version <num>
    python agent_cicd_cli.py compare --agent-id <id> --v1 <num> --v2 <num>
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
parent_dir = Path(__file__).parent.parent / "01_agent"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.evaluation import (
    IntentResolutionEvaluator,
    ToolCallAccuracyEvaluator,
    TaskAdherenceEvaluator,
    AzureOpenAIModelConfiguration,
    AIAgentConverter
)

from agent_db import AgentDB
from agent_utils import AgentManager
from agent_version_manager import AgentVersionManager


class AgentCICDCLI:
    """Command-line interface for agent CI/CD operations"""

    def __init__(self):
        """Initialize the CLI"""
        # Load environment variables
        env_path = Path(__file__).parent.parent / ".env.local"
        if not env_path.exists():
            env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(env_path)

        # Initialize clients
        endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_AI_PROJECT_ENDPOINT not set")

        self.project_client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential()
        )

        self.agent_manager = AgentManager(project_client=self.project_client)
        self.version_manager = AgentVersionManager(agent_db=self.agent_manager.db)

        # Initialize evaluators
        model_config = AzureOpenAIModelConfiguration(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_GPT_4o"],
            api_key=os.environ["AZURE_OPENAI_API_KEY_GPT_4o"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION_GPT_4o"],
            azure_deployment=os.environ["AZURE_OPENAI_MODEl_GPT_4o"],
        )

        self.intent_resolution = IntentResolutionEvaluator(model_config=model_config)
        self.tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=model_config)
        self.task_adherence = TaskAdherenceEvaluator(model_config=model_config)
        self.converter = AIAgentConverter(self.project_client)

    def update_agent(
        self,
        agent_id: str,
        config_file: str,
        change_description: str = None,
        changed_by: str = "cli"
    ) -> Dict[str, Any]:
        """
        Update agent from configuration file

        Args:
            agent_id: Agent ID
            config_file: Path to JSON config file
            change_description: Description of changes
            changed_by: Who made the change

        Returns:
            Result dict
        """
        print(f"📝 Updating agent: {agent_id}")

        # Load config
        with open(config_file, 'r') as f:
            updates = json.load(f)

        # Use description from config if not provided
        if not change_description:
            change_description = updates.get("_change_description", "Update from CLI")
            updates.pop("_change_description", None)

        # Update with versioning
        success = self.version_manager.update_agent_with_versioning(
            agent_id=agent_id,
            updates=updates,
            change_description=change_description,
            changed_by=changed_by
        )

        if success:
            current_version = self.version_manager.get_current_version_number(agent_id)
            print(f"✅ Agent updated to version {current_version}")
            return {"success": True, "version": current_version}
        else:
            print("❌ Update failed")
            return {"success": False, "error": "Update failed"}

    def evaluate_agent(
        self,
        agent_id: str,
        test_queries_file: str,
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate agent with test queries

        Args:
            agent_id: Agent ID
            test_queries_file: Path to JSON file with test queries
            output_file: Optional output file for results

        Returns:
            Evaluation results
        """
        print(f"🔍 Evaluating agent: {agent_id}")

        # Load test queries
        with open(test_queries_file, 'r') as f:
            test_queries = json.load(f)

        if not isinstance(test_queries, list):
            test_queries = test_queries.get("queries", [])

        results = {
            "agent_id": agent_id,
            "version": self.version_manager.get_current_version_number(agent_id),
            "test_results": []
        }

        # Run evaluations
        for query in test_queries:
            print(f"\n  Testing: {query[:60]}...")

            # Create thread and run agent
            thread = self.agent_manager.create_thread()
            message = self.project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
            run = self.project_client.agents.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agent_id
            )

            # Convert and evaluate
            eval_data = self.converter.convert(thread_id=thread.id, run_id=run.id)

            # Run evaluators
            intent_result = self.intent_resolution(
                query=eval_data.get("query"),
                response=eval_data.get("response")
            )
            task_result = self.task_adherence(
                conversation=eval_data.get("conversation", {"messages": []})
            )

            test_result = {
                "query": query,
                "intent_resolution": intent_result.get("intent_resolution"),
                "intent_resolution_pass": intent_result.get("intent_resolution_pass"),
                "task_adherence": task_result.get("task_adherence"),
                "task_adherence_pass": task_result.get("task_adherence_pass")
            }

            results["test_results"].append(test_result)

            # Cleanup
            self.agent_manager.delete_thread(thread.id, silent=True)

        # Calculate summary statistics
        scores = [
            r["intent_resolution"] for r in results["test_results"]
            if r.get("intent_resolution") is not None
        ] + [
            r["task_adherence"] for r in results["test_results"]
            if r.get("task_adherence") is not None
        ]

        results["summary"] = {
            "total_tests": len(test_queries),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0
        }

        print(f"\n✅ Evaluation complete")
        print(f"   Average score: {results['summary']['average_score']:.2f}")

        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"   Results saved to: {output_file}")

        return results

    def rollback_agent(
        self,
        agent_id: str,
        version_number: int,
        change_description: str = None,
        changed_by: str = "cli"
    ) -> Dict[str, Any]:
        """
        Rollback agent to previous version

        Args:
            agent_id: Agent ID
            version_number: Version number to rollback to
            change_description: Description of rollback
            changed_by: Who performed rollback

        Returns:
            Result dict
        """
        print(f"⏮️  Rolling back agent {agent_id} to version {version_number}")

        if not change_description:
            change_description = f"Rollback to version {version_number} via CLI"

        success = self.version_manager.rollback_to_version(
            agent_id=agent_id,
            version_number=version_number,
            change_description=change_description,
            changed_by=changed_by
        )

        if success:
            new_version = self.version_manager.get_current_version_number(agent_id)
            print(f"✅ Rollback successful, now at version {new_version}")
            return {"success": True, "new_version": new_version}
        else:
            print("❌ Rollback failed")
            return {"success": False, "error": "Rollback failed"}

    def compare_versions(
        self,
        agent_id: str,
        version1: int,
        version2: int,
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Compare two versions

        Args:
            agent_id: Agent ID
            version1: First version number
            version2: Second version number
            output_file: Optional output file for comparison

        Returns:
            Comparison results
        """
        print(f"🔍 Comparing version {version1} vs version {version2}")

        comparison = self.version_manager.compare_versions(
            agent_id=agent_id,
            version1=version1,
            version2=version2
        )

        if comparison.get("error"):
            print(f"❌ {comparison['error']}")
            return comparison

        print(f"\n✅ Comparison complete")
        print(f"   Differences found: {len(comparison.get('differences', {}))}")

        if comparison.get("differences"):
            for field, changes in comparison["differences"].items():
                print(f"\n   {field}:")
                print(f"     v{version1}: {str(changes[f'version_{version1}'])[:60]}...")
                print(f"     v{version2}: {str(changes[f'version_{version2}'])[:60]}...")

        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\n   Comparison saved to: {output_file}")

        return comparison

    def version_history(self, agent_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get version history

        Args:
            agent_id: Agent ID
            limit: Maximum number of versions to show

        Returns:
            List of versions
        """
        print(f"📜 Version history for agent: {agent_id}")

        versions = self.version_manager.get_version_history(agent_id, limit=limit)

        print(f"\n   Total versions: {len(versions)}")

        for version in versions:
            print(f"\n   🔖 Version {version['versionNumber']}")
            print(f"      Timestamp: {version['timestamp']}")
            print(f"      Changed by: {version['changedBy']}")
            print(f"      Description: {version['changeDescription']}")

        return versions


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Agent CI/CD CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update agent")
    update_parser.add_argument("--agent-id", required=True, help="Agent ID")
    update_parser.add_argument("--config", required=True, help="Config JSON file")
    update_parser.add_argument("--description", help="Change description")
    update_parser.add_argument("--changed-by", default="cli", help="Who made the change")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate agent")
    eval_parser.add_argument("--agent-id", required=True, help="Agent ID")
    eval_parser.add_argument("--test-queries", required=True, help="Test queries JSON file")
    eval_parser.add_argument("--output", help="Output file for results")

    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback agent")
    rollback_parser.add_argument("--agent-id", required=True, help="Agent ID")
    rollback_parser.add_argument("--version", type=int, required=True, help="Version to rollback to")
    rollback_parser.add_argument("--description", help="Rollback description")
    rollback_parser.add_argument("--changed-by", default="cli", help="Who performed rollback")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare versions")
    compare_parser.add_argument("--agent-id", required=True, help="Agent ID")
    compare_parser.add_argument("--v1", type=int, required=True, help="First version")
    compare_parser.add_argument("--v2", type=int, required=True, help="Second version")
    compare_parser.add_argument("--output", help="Output file for comparison")

    # History command
    history_parser = subparsers.add_parser("history", help="Show version history")
    history_parser.add_argument("--agent-id", required=True, help="Agent ID")
    history_parser.add_argument("--limit", type=int, help="Maximum versions to show")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize CLI
    cli = AgentCICDCLI()

    # Execute command
    if args.command == "update":
        result = cli.update_agent(
            agent_id=args.agent_id,
            config_file=args.config,
            change_description=args.description,
            changed_by=args.changed_by
        )
        sys.exit(0 if result["success"] else 1)

    elif args.command == "evaluate":
        result = cli.evaluate_agent(
            agent_id=args.agent_id,
            test_queries_file=args.test_queries,
            output_file=args.output
        )
        # Exit with error if average score is below 3.0
        sys.exit(0 if result["summary"]["average_score"] >= 3.0 else 1)

    elif args.command == "rollback":
        result = cli.rollback_agent(
            agent_id=args.agent_id,
            version_number=args.version,
            change_description=args.description,
            changed_by=args.changed_by
        )
        sys.exit(0 if result["success"] else 1)

    elif args.command == "compare":
        result = cli.compare_versions(
            agent_id=args.agent_id,
            version1=args.v1,
            version2=args.v2,
            output_file=args.output
        )
        sys.exit(0 if not result.get("error") else 1)

    elif args.command == "history":
        cli.version_history(agent_id=args.agent_id, limit=args.limit)
        sys.exit(0)


if __name__ == "__main__":
    main()
