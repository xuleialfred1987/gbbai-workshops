"""
Redis-based RAG Tools for Voice Assistant

This module provides RAG (Retrieval-Augmented Generation) tools using Azure Managed Redis
with RediSearch for vector search. It supports:
- Vector similarity search using HNSW algorithm
- Hybrid search (vector + metadata filtering)
- Grounding citation reporting

Compatible with the Azure VoiceLive SDK voice assistant.
"""

import json
import logging
import os
from typing import Any, Optional, Dict, Callable, Awaitable

import numpy as np
import redis
from openai import AzureOpenAI
from redis.commands.search.query import Query

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

REDIS_SEARCH_SCHEMA = {
    "type": "function",
    "name": "internal_search",
    "description": (
        "Search the knowledge base for relevant information. "
        "The knowledge base contains product documents that can help answer user questions. "
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
# Redis RAG Tool Implementation
# ==============================================================================

class RedisRAGToolManager:
    """
    Manages Redis-based RAG tools for the voice assistant.
    
    Provides:
    - Azure Managed Redis + RediSearch integration for vector search
    - Grounding/citation tracking
    - Tool handlers for VoiceLive SDK
    """
    
    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: int = 10000,
        redis_password: Optional[str] = None,
        redis_ssl: bool = True,
        index_name: str = "product_index",
        openai_endpoint: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        embedding_dimensions: int = 1536,
        identifier_field: str = "id",
        content_field: str = "content",
        name_field: str = "name",
        metadata_field: str = "metadata",
        embedding_field: str = "embedding",
        top_k: int = 5,
    ):
        """
        Initialize Redis RAG tools.
        
        Args:
            redis_host: Azure Managed Redis host
            redis_port: Redis port (default 10000 for Azure Managed Redis)
            redis_password: Redis access key
            redis_ssl: Whether to use SSL (default True)
            index_name: RediSearch index name
            openai_endpoint: Azure OpenAI endpoint for embeddings
            openai_api_key: Azure OpenAI API key
            embedding_model: Embedding model deployment name
            embedding_dimensions: Embedding vector dimensions
            identifier_field: Field name for document ID
            content_field: Field name for document content
            name_field: Field name for document name/title
            metadata_field: Field name for document metadata
            embedding_field: Field name for vector embeddings
            top_k: Default number of results to return
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.redis_ssl = redis_ssl
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.identifier_field = identifier_field
        self.content_field = content_field
        self.name_field = name_field
        self.metadata_field = metadata_field
        self.embedding_field = embedding_field
        self.top_k = top_k
        
        # Track grounding sources
        self.grounding_sources: list[str] = []
        
        # Initialize Redis client
        self.redis_client: Optional[redis.Redis] = None
        if redis_host and redis_password:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                    ssl=redis_ssl,
                    ssl_cert_reqs=None,  # For Azure Managed Redis
                    decode_responses=False,  # Vector data needs bytes
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"Redis RAG client initialized: {redis_host}:{redis_port}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis_client = None
        else:
            logger.warning("Redis configuration incomplete - search will return errors")
        
        # Initialize OpenAI client for embeddings
        self.openai_client: Optional[AzureOpenAI] = None
        if openai_endpoint and openai_api_key:
            self.openai_client = AzureOpenAI(
                api_key=openai_api_key,
                api_version="2024-02-01",
                azure_endpoint=openai_endpoint
            )
            logger.info("OpenAI client initialized for embeddings")
        else:
            logger.warning("OpenAI configuration incomplete - embeddings will fail")
    
    def get_tool_schemas(self) -> list[dict]:
        """Get all RAG tool schemas for VoiceLive configuration."""
        return [REDIS_SEARCH_SCHEMA, GROUNDING_REPORT_SCHEMA]
    
    def get_tool_handlers(self) -> Dict[str, Callable[[Any], Awaitable[ToolResult]]]:
        """Get tool handlers for VoiceLive."""
        return {
            "internal_search": self._search_handler,
            "report_grounding": self._grounding_handler,
        }
    
    def _generate_embedding(self, text: str) -> bytes:
        """Generate embedding for query text and return as bytes."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        response = self.openai_client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32).tobytes()
    
    async def _search_handler(self, args: Any) -> ToolResult:
        """Handle internal_search tool calls using Redis vector search."""
        query = (args.get("query") or "").strip()
        
        if not query:
            return ToolResult(
                json.dumps({"error": "Query must be provided.", "chunks": [], "count": 0}),
                ToolResultDirection.TO_SERVER
            )
        
        if not self.redis_client:
            return ToolResult(
                json.dumps({
                    "error": "Redis not configured",
                    "details": "Set AZURE_REDIS_HOST and AZURE_REDIS_PASSWORD"
                }),
                ToolResultDirection.TO_SERVER
            )
        
        if not self.openai_client:
            return ToolResult(
                json.dumps({
                    "error": "OpenAI not configured",
                    "details": "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
                }),
                ToolResultDirection.TO_SERVER
            )
        
        logger.info(f"Redis searching for: {query}")
        print(f"🔍 Searching Redis knowledge base for: {query}")
        
        try:
            # Generate query embedding
            query_vector = self._generate_embedding(query)
            
            # Build KNN vector search query
            # Syntax: *=>[KNN K @field $param AS score]
            redis_query = (
                Query(f"*=>[KNN {self.top_k} @{self.embedding_field} $query_vec AS score]")
                .sort_by("score")
                .return_fields(
                    self.identifier_field, 
                    self.name_field, 
                    self.content_field, 
                    self.metadata_field, 
                    "score"
                )
                .dialect(2)
            )
            
            # Execute search
            results = self.redis_client.ft(self.index_name).search(
                redis_query,
                query_params={"query_vec": query_vector}
            )
            
            # Process results
            chunks: list[dict] = []
            formatted_sections: list[str] = []
            
            for doc in results.docs:
                # Decode bytes to strings
                chunk_id = doc.id if hasattr(doc, 'id') else getattr(doc, self.identifier_field, "unknown")
                if isinstance(chunk_id, bytes):
                    chunk_id = chunk_id.decode()
                
                name = getattr(doc, self.name_field, "")
                if isinstance(name, bytes):
                    name = name.decode()
                
                content = getattr(doc, self.content_field, "")
                if isinstance(content, bytes):
                    content = content.decode()
                
                metadata = getattr(doc, self.metadata_field, "")
                if isinstance(metadata, bytes):
                    metadata = metadata.decode()
                
                score = float(doc.score) if hasattr(doc, 'score') else 0
                similarity = 1 - score  # COSINE distance to similarity
                
                chunk = {
                    "id": chunk_id,
                    "name": name,
                    "content": content,
                    "metadata": metadata,
                    "score": score,
                    "similarity": similarity,
                }
                chunks.append(chunk)
                
                # Format for LLM context: use name as identifier
                source_id = name if name else chunk_id
                formatted_sections.append(f"[{source_id}]: {content}\n{metadata}\n-----\n")
            
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
                for i, chunk in enumerate(chunks, 1):
                    print(f"      {i}. {chunk['name']} (similarity: {chunk['similarity']:.4f})")
            
            return ToolResult(json.dumps(payload, ensure_ascii=False), ToolResultDirection.TO_SERVER)
            
        except Exception as e:
            logger.exception("Redis search failed")
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
        """Close the Redis client."""
        if self.redis_client:
            self.redis_client.close()


def create_redis_rag_tools_from_env() -> RedisRAGToolManager:
    """
    Create Redis RAG tools using environment variables.
    
    Environment variables:
    - AZURE_REDIS_HOST: Azure Managed Redis host
    - AZURE_REDIS_PORT: Redis port (default: 10000)
    - AZURE_REDIS_PASSWORD: Redis access key
    - AZURE_REDIS_SSL: Use SSL (default: "true")
    - AZURE_REDIS_INDEX_NAME: RediSearch index name (default: "product_index")
    - AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint
    - AZURE_OPENAI_API_KEY: Azure OpenAI API key
    - AZURE_OPENAI_EMBEDDING_MODEL: Embedding model (default: "text-embedding-ada-002")
    - AZURE_REDIS_TOP_K: Number of results (default: 5)
    """
    return RedisRAGToolManager(
        redis_host=os.environ.get("AZURE_REDIS_HOST"),
        redis_port=int(os.environ.get("AZURE_REDIS_PORT", "10000")),
        redis_password=os.environ.get("AZURE_REDIS_PASSWORD"),
        redis_ssl=os.environ.get("AZURE_REDIS_SSL", "true").lower() == "true",
        index_name=os.environ.get("AZURE_REDIS_INDEX_NAME", "product_index"),
        openai_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        embedding_model=os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        top_k=int(os.environ.get("AZURE_REDIS_TOP_K", "5")),
    )


# ==============================================================================
# Synchronous wrapper for non-async usage
# ==============================================================================

class RedisRAGToolManagerSync:
    """
    Synchronous version of RedisRAGToolManager for use in non-async contexts.
    """
    
    def __init__(self, async_manager: RedisRAGToolManager):
        self._manager = async_manager
    
    def search(self, query: str) -> dict:
        """
        Execute a vector search query.
        
        Args:
            query: Search query text
            
        Returns:
            Dict with query results
        """
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            self._manager._search_handler({"query": query})
        )
        return json.loads(result.text)
    
    def get_tool_schemas(self) -> list[dict]:
        return self._manager.get_tool_schemas()
    
    def close(self):
        if self._manager.redis_client:
            self._manager.redis_client.close()


# Export
__all__ = [
    "ToolResult",
    "ToolResultDirection",
    "RedisRAGToolManager",
    "RedisRAGToolManagerSync",
    "create_redis_rag_tools_from_env",
    "REDIS_SEARCH_SCHEMA",
    "GROUNDING_REPORT_SCHEMA",
]
