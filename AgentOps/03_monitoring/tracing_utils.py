"""
Azure OpenAI Tracing Utilities

Reusable tracing functions for monitoring Azure OpenAI API calls with OpenTelemetry.
Supports both synchronous and asynchronous operations, streaming and non-streaming modes,
and persistent storage in SQLite.
"""

import time
import sqlite3
from typing import List, Dict, Any, Tuple
from openai import AzureOpenAI, AsyncAzureOpenAI
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanExportResult
)


class SQLiteSpanExporter(SpanExporter):
    """Custom OpenTelemetry exporter that stores spans in SQLite database"""

    def __init__(self, db_path: str = "azure_openai_traces.db", verbose: bool = False):
        """
        Initialize SQLite span exporter

        Args:
            db_path: Path to SQLite database file
            verbose: Enable verbose logging
        """
        self.db_path = db_path
        self.verbose = verbose
        self._init_database()

    def _init_database(self):
        """Create the database schema if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create spans table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT,
                start_time INTEGER,
                end_time INTEGER,
                duration_ms REAL,
                status_code TEXT,
                status_description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create attributes table with CASCADE DELETE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS span_attributes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                span_id INTEGER,
                key TEXT NOT NULL,
                value TEXT,
                FOREIGN KEY (span_id) REFERENCES spans(id) ON DELETE CASCADE
            )
        """)

        # Create indexes for faster queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_trace_id ON spans(trace_id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_span_name ON spans(name)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON spans(created_at)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_attr_key ON span_attributes(key)")

        conn.commit()
        conn.close()

        if self.verbose:
            print(f"✅ SQLite database initialized: {self.db_path}")

    def export(self, spans) -> SpanExportResult:
        """Export spans to SQLite database"""
        if self.verbose:
            print(f"📤 Exporting {len(spans)} span(s) to database...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            for span in spans:
                # Insert span
                trace_id = format(span.context.trace_id, '032x')
                span_id = format(span.context.span_id, '016x')
                duration_ms = (span.end_time - span.start_time) / 1_000_000

                cursor.execute("""
                    INSERT INTO spans (
                        trace_id, span_id, name, kind, start_time, end_time, 
                        duration_ms, status_code, status_description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace_id,
                    span_id,
                    span.name,
                    str(span.kind),
                    span.start_time,
                    span.end_time,
                    duration_ms,
                    str(span.status.status_code),
                    span.status.description
                ))

                db_span_id = cursor.lastrowid

                # Insert attributes
                for key, value in span.attributes.items():
                    cursor.execute("""
                        INSERT INTO span_attributes (span_id, key, value)
                        VALUES (?, ?, ?)
                    """, (db_span_id, key, str(value)))

            conn.commit()

            if self.verbose:
                print(f"✅ Successfully exported {len(spans)} span(s)")

            return SpanExportResult.SUCCESS

        except Exception as e:
            print(f"❌ Error exporting spans to SQLite: {e}")
            conn.rollback()
            return SpanExportResult.FAILURE
        finally:
            conn.close()

    def shutdown(self):
        """Shutdown the exporter"""
        pass


def setup_tracing(
    service_name: str = "azure-openai-service",
    enable_console: bool = False,
    enable_sqlite: bool = True,
    sqlite_db_path: str = "azure_openai_traces.db",
    use_batch_processor: bool = True,
    verbose: bool = False
) -> Tracer:
    """
    Initialize OpenTelemetry tracing with multiple exporters

    Args:
        service_name: Name of the service for tracing
        enable_console: Enable console output of traces
        enable_sqlite: Enable SQLite storage of traces
        sqlite_db_path: Path to SQLite database
        use_batch_processor: Use BatchSpanProcessor (True) or SimpleSpanProcessor (False)
        verbose: Enable verbose logging

    Returns:
        Configured tracer instance
    """
    # Create tracer provider
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    # Add console exporter if enabled
    if enable_console:
        console_exporter = ConsoleSpanExporter()
        processor_class = BatchSpanProcessor if use_batch_processor else SimpleSpanProcessor
        console_processor = processor_class(console_exporter)
        provider.add_span_processor(console_processor)
        if verbose:
            print("✅ Console tracing enabled")

    # Add SQLite exporter if enabled
    if enable_sqlite:
        sqlite_exporter = SQLiteSpanExporter(
            db_path=sqlite_db_path, verbose=verbose)
        processor_class = BatchSpanProcessor if use_batch_processor else SimpleSpanProcessor
        sqlite_processor = processor_class(sqlite_exporter)
        provider.add_span_processor(sqlite_processor)
        if verbose:
            print(f"✅ SQLite tracing enabled: {sqlite_db_path}")

    # Get tracer
    tracer = trace.get_tracer(service_name)

    if verbose:
        print(f"✅ Tracing initialized for service: {service_name}")

    return tracer


def traced_chat_completion(
    client: AzureOpenAI,
    tracer: Tracer,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 500,
    **kwargs
) -> Tuple[Any, float]:
    """
    Synchronous Azure OpenAI chat completion with tracing

    Args:
        client: AzureOpenAI client instance
        tracer: OpenTelemetry tracer
        messages: List of message dictionaries
        model: Model deployment name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters for chat.completions.create

    Returns:
        Tuple of (response, latency_seconds)
    """
    with tracer.start_as_current_span(
        "azure_openai_chat_completion",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.system": "azure.openai",
            "gen_ai.request.model": model,
            "gen_ai.operation.name": "chat.completions",
            "gen_ai.request.temperature": temperature,
            "gen_ai.request.max_tokens": max_tokens,
        }
    ) as span:
        start_time = time.time()

        try:
            # Add message attributes
            for idx, msg in enumerate(messages):
                span.set_attribute(f"gen_ai.prompt.{idx}.role", msg["role"])
                span.set_attribute(
                    f"gen_ai.prompt.{idx}.content", msg["content"])

            # Make API call
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Calculate latency
            latency = time.time() - start_time

            # Add response attributes
            span.set_attribute("gen_ai.response.id", response.id)
            span.set_attribute("gen_ai.response.model", response.model)
            span.set_attribute("gen_ai.response.finish_reason",
                               response.choices[0].finish_reason)
            span.set_attribute("gen_ai.completion.0.role",
                               response.choices[0].message.role)
            span.set_attribute("gen_ai.completion.0.content",
                               response.choices[0].message.content)

            # Token usage
            if response.usage:
                span.set_attribute("gen_ai.usage.prompt_tokens",
                                   response.usage.prompt_tokens)
                span.set_attribute(
                    "gen_ai.usage.completion_tokens", response.usage.completion_tokens)
                span.set_attribute("gen_ai.usage.total_tokens",
                                   response.usage.total_tokens)

            # Performance metrics
            span.set_attribute("gen_ai.response.latency_ms", latency * 1000)
            span.set_status(Status(StatusCode.OK))

            return response, latency

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


async def traced_chat_completion_async(
    client: AsyncAzureOpenAI,
    tracer: Tracer,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 500,
    **kwargs
) -> Tuple[Any, float]:
    """
    Asynchronous Azure OpenAI chat completion with tracing

    Args:
        client: AsyncAzureOpenAI client instance
        tracer: OpenTelemetry tracer
        messages: List of message dictionaries
        model: Model deployment name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters for chat.completions.create

    Returns:
        Tuple of (response, latency_seconds)
    """
    with tracer.start_as_current_span(
        "azure_openai_async_chat_completion",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.system": "azure.openai",
            "gen_ai.request.model": model,
            "gen_ai.operation.name": "chat.completions",
            "gen_ai.request.temperature": temperature,
            "gen_ai.request.max_tokens": max_tokens,
        }
    ) as span:
        start_time = time.time()

        try:
            # Add message attributes
            for idx, msg in enumerate(messages):
                span.set_attribute(f"gen_ai.prompt.{idx}.role", msg["role"])
                span.set_attribute(
                    f"gen_ai.prompt.{idx}.content", msg["content"])

            # Make async API call
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Calculate latency
            latency = time.time() - start_time

            # Add response attributes
            span.set_attribute("gen_ai.response.id", response.id)
            span.set_attribute("gen_ai.response.model", response.model)
            span.set_attribute("gen_ai.response.finish_reason",
                               response.choices[0].finish_reason)
            span.set_attribute("gen_ai.completion.0.role",
                               response.choices[0].message.role)
            span.set_attribute("gen_ai.completion.0.content",
                               response.choices[0].message.content)

            # Token usage
            if response.usage:
                span.set_attribute("gen_ai.usage.prompt_tokens",
                                   response.usage.prompt_tokens)
                span.set_attribute(
                    "gen_ai.usage.completion_tokens", response.usage.completion_tokens)
                span.set_attribute("gen_ai.usage.total_tokens",
                                   response.usage.total_tokens)

            # Performance metrics
            span.set_attribute("gen_ai.response.latency_ms", latency * 1000)
            span.set_status(Status(StatusCode.OK))

            return response, latency

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def traced_chat_completion_streaming(
    client: AzureOpenAI,
    tracer: Tracer,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 500,
    **kwargs
) -> Tuple[str, int, float]:
    """
    Synchronous Azure OpenAI streaming chat completion with tracing

    Args:
        client: AzureOpenAI client instance
        tracer: OpenTelemetry tracer
        messages: List of message dictionaries
        model: Model deployment name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters for chat.completions.create

    Returns:
        Tuple of (full_content, chunk_count, latency_seconds)
    """
    with tracer.start_as_current_span(
        "azure_openai_streaming",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.system": "azure.openai",
            "gen_ai.request.model": model,
            "gen_ai.operation.name": "chat.completions.streaming",
            "gen_ai.request.streaming": True,
            "gen_ai.request.temperature": temperature,
            "gen_ai.request.max_tokens": max_tokens,
        }
    ) as span:
        start_time = time.time()

        try:
            # Add message attributes
            for idx, msg in enumerate(messages):
                span.set_attribute(f"gen_ai.prompt.{idx}.role", msg["role"])
                span.set_attribute(
                    f"gen_ai.prompt.{idx}.content", msg["content"])

            # Make streaming API call
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            # Collect streamed content
            full_content = ""
            chunk_count = 0

            for chunk in stream:
                chunk_count += 1
                if chunk.choices and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content

            # Calculate metrics
            latency = time.time() - start_time
            estimated_tokens = len(full_content.split())

            # Add final attributes
            span.set_attribute("gen_ai.completion.0.content", full_content)
            span.set_attribute("gen_ai.response.chunk_count", chunk_count)
            span.set_attribute("gen_ai.response.latency_ms", latency * 1000)
            span.set_attribute(
                "gen_ai.usage.estimated_completion_tokens", estimated_tokens)
            span.set_status(Status(StatusCode.OK))

            return full_content, chunk_count, latency

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


async def traced_chat_completion_streaming_async(
    client: AsyncAzureOpenAI,
    tracer: Tracer,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 500,
    **kwargs
) -> Tuple[str, int, float]:
    """
    Asynchronous Azure OpenAI streaming chat completion with tracing

    Args:
        client: AsyncAzureOpenAI client instance
        tracer: OpenTelemetry tracer
        messages: List of message dictionaries
        model: Model deployment name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters for chat.completions.create

    Returns:
        Tuple of (full_content, chunk_count, latency_seconds)
    """
    with tracer.start_as_current_span(
        "azure_openai_async_streaming",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.system": "azure.openai",
            "gen_ai.request.model": model,
            "gen_ai.operation.name": "chat.completions.streaming",
            "gen_ai.request.streaming": True,
            "gen_ai.request.temperature": temperature,
            "gen_ai.request.max_tokens": max_tokens,
        }
    ) as span:
        start_time = time.time()

        try:
            # Add message attributes
            for idx, msg in enumerate(messages):
                span.set_attribute(f"gen_ai.prompt.{idx}.role", msg["role"])
                span.set_attribute(
                    f"gen_ai.prompt.{idx}.content", msg["content"])

            # Make async streaming API call
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            # Collect streamed content
            full_content = ""
            chunk_count = 0

            async for chunk in stream:
                chunk_count += 1
                if chunk.choices and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content

            # Calculate metrics
            latency = time.time() - start_time
            estimated_tokens = len(full_content.split())

            # Add final attributes
            span.set_attribute("gen_ai.completion.0.content", full_content)
            span.set_attribute("gen_ai.response.chunk_count", chunk_count)
            span.set_attribute("gen_ai.response.latency_ms", latency * 1000)
            span.set_attribute(
                "gen_ai.usage.estimated_completion_tokens", estimated_tokens)
            span.set_status(Status(StatusCode.OK))

            return full_content, chunk_count, latency

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def flush_traces(timeout: int = 30):
    """
    Force flush all pending traces to exporters

    Args:
        timeout: Timeout in seconds
    """
    trace.get_tracer_provider().force_flush(timeout_millis=timeout * 1000)
