"""
RAG Tools for Voice Assistant

This module provides RAG (Retrieval-Augmented Generation) tools for use with 
the Azure VoiceLive SDK voice assistant. It supports:
- Azure AI Search with semantic ranking
- Vector search using embeddings
- Grounding citation reporting

Based on the backend tools.py but adapted for notebook use.
"""

import json
import logging
import os
from typing import Any, Optional, Dict, Callable, Awaitable

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.aio import SearchClient

# VectorizableTextQuery is available in azure-search-documents >= 11.5.0
# For older versions, use VectorizedQuery or skip vector search
try:
    from azure.search.documents.models import VectorizableTextQuery
    HAS_VECTORIZABLE_TEXT_QUERY = True
except ImportError:
    try:
        from azure.search.documents._generated.models import VectorizableTextQuery
        HAS_VECTORIZABLE_TEXT_QUERY = True
    except ImportError:
        HAS_VECTORIZABLE_TEXT_QUERY = False
        VectorizableTextQuery = None

logger = logging.getLogger(__name__)


# ==============================================================================
# Tool Result Types
# ==============================================================================

class ToolResultDirection:
    """Direction for tool results."""
    TO_SERVER = 1  # Send result to the LLM
    TO_CLIENT = 2  # Send result to the client UI


class ToolResult:
    """Result from a tool execution."""

    def __init__(self, text: str, destination: int = ToolResultDirection.TO_SERVER):
        self.text = text
        self.destination = destination

    def to_text(self) -> str:
        if self.text is None:
            return ""
        return self.text if isinstance(self.text, str) else json.dumps(self.text)


# ==============================================================================
# Tool Schemas
# ==============================================================================

INTERNAL_SEARCH_SCHEMA = {
    "type": "function",
    "name": "internal_search",
    "description": (
        "Search the knowledge base for relevant information. "
        "The knowledge base contains documents that can help answer user questions. "
        "Results are formatted with a source identifier in square brackets, followed by "
        "the content, and a line with '-----' separating each result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant information"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

GROUNDING_REPORT_SCHEMA = {
    "type": "function",
    "name": "report_grounding",
    "description": (
        "Report use of a source from the knowledge base as part of an answer. "
        "Sources appear in square brackets before each knowledge base passage. "
        "Always use this tool to cite sources when responding with information from the knowledge base."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of source names actually used in the response"
            }
        },
        "required": ["sources"],
        "additionalProperties": False
    }
}


# ==============================================================================
# RAG Tool Implementation
# ==============================================================================

class RAGToolManager:
    """
    Manages RAG tools for the voice assistant.

    Provides:
    - Azure AI Search integration with semantic and vector search
    - Grounding/citation tracking
    - Tool handlers for VoiceLive SDK
    """

    def __init__(
        self,
        search_endpoint: Optional[str] = None,
        search_index: Optional[str] = None,
        search_credential: Optional[AzureKeyCredential |
                                    DefaultAzureCredential] = None,
        semantic_configuration: str = "default",
        identifier_field: str = "id",
        content_field: str = "content",
        embedding_field: str = "embedding",
        title_field: Optional[str] = None,
        use_vector_query: bool = True,
    ):
        """
        Initialize RAG tools.

        Args:
            search_endpoint: Azure AI Search endpoint URL
            search_index: Name of the search index
            search_credential: Azure credential for search
            semantic_configuration: Semantic search configuration name
            identifier_field: Field name for document ID
            content_field: Field name for document content
            embedding_field: Field name for vector embeddings
            title_field: Optional field name for document title
            use_vector_query: Whether to use vector search
        """
        self.search_endpoint = search_endpoint
        self.search_index = search_index
        self.search_credential = search_credential
        self.semantic_configuration = semantic_configuration
        self.identifier_field = identifier_field
        self.content_field = content_field
        self.embedding_field = embedding_field
        self.title_field = title_field
        self.use_vector_query = use_vector_query

        # Track grounding sources
        self.grounding_sources: list[str] = []

        # Initialize search client if configured
        self.search_client: Optional[SearchClient] = None
        if search_endpoint and search_index and search_credential:
            self.search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=search_index,
                credential=search_credential,
            )
            logger.info(
                f"RAG search client initialized for index: {search_index}")
        else:
            logger.warning(
                "Search configuration incomplete - search will return errors")

    def get_tool_schemas(self) -> list[dict]:
        """Get all RAG tool schemas for VoiceLive configuration."""
        return [INTERNAL_SEARCH_SCHEMA, GROUNDING_REPORT_SCHEMA]

    def get_tool_handlers(self) -> Dict[str, Callable[[Any], Awaitable[ToolResult]]]:
        """Get tool handlers for VoiceLive."""
        return {
            "internal_search": self._search_handler,
            "report_grounding": self._grounding_handler,
        }

    async def _search_handler(self, args: Any) -> ToolResult:
        """Handle internal_search tool calls."""
        query = (args.get("query") or "").strip()

        if not query:
            return ToolResult(
                json.dumps({"error": "Query must be provided.",
                           "chunks": [], "count": 0}),
                ToolResultDirection.TO_SERVER
            )

        if not self.search_client:
            return ToolResult(
                json.dumps({
                    "error": "Search not configured",
                    "details": "Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX"
                }),
                ToolResultDirection.TO_SERVER
            )

        logger.info(f"Searching for: {query}")
        print(f"🔍 Searching knowledge base for: {query}")

        try:
            # Build vector query if enabled and available
            vector_queries = []
            if self.use_vector_query and HAS_VECTORIZABLE_TEXT_QUERY and VectorizableTextQuery:
                vector_queries.append(VectorizableTextQuery(
                    text=query,
                    k_nearest_neighbors=50,
                    fields=self.embedding_field
                ))
            elif self.use_vector_query and not HAS_VECTORIZABLE_TEXT_QUERY:
                logger.warning(
                    "VectorizableTextQuery not available, falling back to semantic-only search")

            # Build select fields
            select_fields = [self.identifier_field, self.content_field]
            if self.title_field and self.title_field not in select_fields:
                select_fields.append(self.title_field)

            # Execute search
            search_results = await self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name=self.semantic_configuration,
                top=3,
                vector_queries=vector_queries if vector_queries else None,
                select=", ".join(select_fields)
            )

            # Process results
            chunks: list[dict] = []
            formatted_sections: list[str] = []

            async for result in search_results:
                chunk_id = result.get(self.identifier_field)
                chunk_content = result.get(self.content_field)

                chunk = {
                    "id": chunk_id,
                    "content": chunk_content,
                    "score": result.get("@search.score"),
                    "reranker_score": result.get("@search.reranker_score"),
                }
                if self.title_field:
                    chunk["title"] = result.get(self.title_field)

                chunks.append(chunk)
                formatted_sections.append(
                    f"[{chunk_id}]: {chunk_content}\n-----\n")

            results_text = "".join(formatted_sections).strip()

            payload = {
                "query": query,
                "results_text": results_text,
                "chunks": chunks,
                "count": len(chunks),
            }

            if not chunks:
                payload["message"] = "No results found."
                print("   ⚠️ No results found")
            else:
                print(f"   ✅ Found {len(chunks)} results")

            return ToolResult(json.dumps(payload, ensure_ascii=False), ToolResultDirection.TO_SERVER)

        except Exception as e:
            logger.exception("Search failed")
            return ToolResult(
                json.dumps({"error": str(e), "chunks": [], "count": 0}),
                ToolResultDirection.TO_SERVER
            )

    async def _grounding_handler(self, args: Any) -> ToolResult:
        """Handle report_grounding tool calls."""
        sources = args.get("sources", [])

        if sources:
            self.grounding_sources.extend(sources)
            print(f"📚 Cited sources: {', '.join(sources)}")
            logger.info(f"Grounding sources cited: {sources}")

        return ToolResult(
            json.dumps({"status": "ok", "sources_recorded": sources}),
            ToolResultDirection.TO_SERVER
        )

    def get_cited_sources(self) -> list[str]:
        """Get all sources cited during the session."""
        return list(set(self.grounding_sources))

    def clear_citations(self):
        """Clear the citation list."""
        self.grounding_sources.clear()

    async def close(self):
        """Close the search client."""
        if self.search_client:
            await self.search_client.close()


def create_rag_tools_from_env() -> RAGToolManager:
    """
    Create RAG tools using environment variables.

    Environment variables:
    - AZURE_SEARCH_ENDPOINT: Azure AI Search endpoint
    - AZURE_SEARCH_INDEX: Search index name
    - AZURE_SEARCH_API_KEY: API key (optional, uses DefaultAzureCredential if not set)
    - AZURE_SEARCH_SEMANTIC_CONFIGURATION: Semantic config name (default: "default")
    - AZURE_SEARCH_IDENTIFIER_FIELD: ID field name (default: "id")
    - AZURE_SEARCH_CONTENT_FIELD: Content field name (default: "content")
    - AZURE_SEARCH_EMBEDDING_FIELD: Embedding field name (default: "embedding")
    - AZURE_SEARCH_TITLE_FIELD: Title field name (optional)
    - AZURE_SEARCH_USE_VECTOR: Use vector search (default: "true")
    """
    search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    search_index = os.environ.get("AZURE_SEARCH_INDEX")
    search_key = os.environ.get("AZURE_SEARCH_API_KEY")

    # Create credential
    if search_key:
        credential = AzureKeyCredential(search_key)
    else:
        credential = DefaultAzureCredential()

    return RAGToolManager(
        search_endpoint=search_endpoint,
        search_index=search_index,
        search_credential=credential,
        semantic_configuration=os.environ.get(
            "AZURE_SEARCH_SEMANTIC_CONFIGURATION", "default"),
        identifier_field=os.environ.get("AZURE_SEARCH_IDENTIFIER_FIELD", "id"),
        content_field=os.environ.get("AZURE_SEARCH_CONTENT_FIELD", "content"),
        embedding_field=os.environ.get(
            "AZURE_SEARCH_EMBEDDING_FIELD", "embedding"),
        title_field=os.environ.get("AZURE_SEARCH_TITLE_FIELD"),
        use_vector_query=os.environ.get(
            "AZURE_SEARCH_USE_VECTOR", "true").lower() == "true",
    )


# Export
__all__ = [
    "ToolResult",
    "ToolResultDirection",
    "RAGToolManager",
    "create_rag_tools_from_env",
    "INTERNAL_SEARCH_SCHEMA",
    "GROUNDING_REPORT_SCHEMA",
]
