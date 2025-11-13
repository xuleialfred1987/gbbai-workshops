"""
OpenTelemetry Tracing Decorators for Azure OpenAI

This module provides decorators to add tracing to existing functions non-intrusively.
Simply decorate your functions to automatically capture traces with rich metadata.
"""

import time
import sqlite3
import inspect
import functools
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from opentelemetry import trace
from opentelemetry.trace import Tracer, SpanKind, Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.resources import Resource


# ============================================================================
# SQLite Exporter
# ============================================================================

class SQLiteSpanExporter:
    """Export spans to SQLite database"""

    def __init__(self, db_path: str = "traces.db", verbose: bool = False):
        self.db_path = db_path
        self.verbose = verbose
        self.service_name = None  # Will be set by setup_tracing
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create spans table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                parent_span_id TEXT,
                name TEXT NOT NULL,
                kind TEXT,
                service_name TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                duration_ms REAL,
                status_code TEXT,
                status_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create span_attributes table with CASCADE DELETE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS span_attributes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                span_id INTEGER,
                key TEXT NOT NULL,
                value TEXT,
                FOREIGN KEY (span_id) REFERENCES spans(id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        conn.close()

        if self.verbose:
            print(f"✅ SQLite database initialized: {self.db_path}")

    def export(self, spans):
        """Export spans to SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for span in spans:
            # Calculate duration
            duration_ms = (span.end_time - span.start_time) / 1_000_000

            # Get service name from span resource attributes if available
            service_name = self.service_name
            if span.resource and span.resource.attributes:
                service_name = span.resource.attributes.get('service.name', self.service_name)

            # Insert span
            cursor.execute("""
                INSERT INTO spans (
                    trace_id, span_id, parent_span_id, name, kind, service_name,
                    start_time, end_time, duration_ms, status_code, status_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                format(span.context.trace_id, '032x'),
                format(span.context.span_id, '016x'),
                format(span.parent.span_id, '016x') if span.parent else None,
                span.name,
                span.kind.name if span.kind else None,
                service_name,
                datetime.fromtimestamp(span.start_time / 1e9).isoformat(),
                datetime.fromtimestamp(span.end_time / 1e9).isoformat(),
                duration_ms,
                span.status.status_code.name if span.status else None,
                span.status.description if span.status else None,
            ))

            span_db_id = cursor.lastrowid

            # Insert attributes (including service.name for OpenTelemetry compatibility)
            if span.attributes:
                for key, value in span.attributes.items():
                    cursor.execute("""
                        INSERT INTO span_attributes (span_id, key, value)
                        VALUES (?, ?, ?)
                    """, (span_db_id, key, str(value)))
            
            # Also store service.name as attribute for OTel compatibility
            if service_name:
                cursor.execute("""
                    INSERT INTO span_attributes (span_id, key, value)
                    VALUES (?, ?, ?)
                """, (span_db_id, 'service.name', service_name))

        conn.commit()
        conn.close()

        return trace.SpanExportResult.SUCCESS

    def shutdown(self):
        """Shutdown exporter"""
        pass


# ============================================================================
# Tracer Setup
# ============================================================================

_tracer: Optional[Tracer] = None


def setup_tracing(
    service_name: str = "azure-openai-service",
    enable_console: bool = False,
    enable_sqlite: bool = True,
    sqlite_db_path: str = "traces.db",
    use_batch_processor: bool = True,
    verbose: bool = False
) -> Tracer:
    """
    Initialize OpenTelemetry tracing

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
    global _tracer

    # Create resource with service name
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    if enable_console:
        console_exporter = ConsoleSpanExporter()
        processor_class = BatchSpanProcessor if use_batch_processor else SimpleSpanProcessor
        console_processor = processor_class(console_exporter)
        provider.add_span_processor(console_processor)
        if verbose:
            print("✅ Console tracing enabled")

    if enable_sqlite:
        sqlite_exporter = SQLiteSpanExporter(
            db_path=sqlite_db_path, verbose=verbose)
        sqlite_exporter.service_name = service_name  # Set the service name
        processor_class = BatchSpanProcessor if use_batch_processor else SimpleSpanProcessor
        sqlite_processor = processor_class(sqlite_exporter)
        provider.add_span_processor(sqlite_processor)
        if verbose:
            print(f"✅ SQLite tracing enabled: {sqlite_db_path}")

    _tracer = trace.get_tracer(service_name)

    if verbose:
        print(f"✅ Tracing initialized for service: {service_name}")

    return _tracer


def get_tracer() -> Tracer:
    """Get the global tracer instance"""
    if _tracer is None:
        return setup_tracing()
    return _tracer


def flush_traces(timeout: int = 10):
    """Force flush all pending traces"""
    provider = trace.get_tracer_provider()
    if hasattr(provider, 'force_flush'):
        provider.force_flush(timeout_millis=timeout * 1000)


# ============================================================================
# Decorator: Generic Function Tracing
# ============================================================================

def trace_function(
    span_name: Optional[str] = None,
    span_kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    capture_args: bool = True,
    capture_result: bool = True,
):
    """
    Decorator to add tracing to any function

    Args:
        span_name: Custom span name (defaults to function name)
        span_kind: OpenTelemetry span kind
        attributes: Additional attributes to add to span
        capture_args: Whether to capture function arguments
        capture_result: Whether to capture function return value

    Example:
        @trace_function(span_name="process_data", attributes={"version": "1.0"})
        def process_data(data):
            return data.upper()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = span_name or f"{func.__module__}.{func.__name__}"

            with tracer.start_as_current_span(
                name,
                kind=span_kind,
                attributes=attributes or {}
            ) as span:
                start_time = time.time()

                try:
                    # Capture arguments
                    if capture_args:
                        sig = inspect.signature(func)
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()

                        for param_name, param_value in bound_args.arguments.items():
                            # Capture simple types directly
                            if isinstance(param_value, (str, int, float, bool)):
                                span.set_attribute(
                                    f"arg.{param_name}", param_value)
                            # Serialize dict and list as JSON
                            elif isinstance(param_value, (dict, list)):
                                import json
                                try:
                                    span.set_attribute(
                                        f"arg.{param_name}", json.dumps(param_value))
                                except (TypeError, ValueError):
                                    # If not JSON serializable, just store type
                                    span.set_attribute(
                                        f"arg.{param_name}.type", type(param_value).__name__)
                            else:
                                span.set_attribute(
                                    f"arg.{param_name}.type", type(param_value).__name__)

                    # Execute function
                    result = func(*args, **kwargs)

                    # Calculate latency
                    latency = time.time() - start_time
                    span.set_attribute("latency_ms", latency * 1000)

                    # Capture result
                    if capture_result and result is not None:
                        if isinstance(result, (str, int, float, bool)):
                            span.set_attribute("result", result)
                        else:
                            span.set_attribute(
                                "result.type", type(result).__name__)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = span_name or f"{func.__module__}.{func.__name__}"

            with tracer.start_as_current_span(
                name,
                kind=span_kind,
                attributes=attributes or {}
            ) as span:
                start_time = time.time()

                try:
                    # Capture arguments
                    if capture_args:
                        sig = inspect.signature(func)
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()

                        for param_name, param_value in bound_args.arguments.items():
                            # Capture simple types directly
                            if isinstance(param_value, (str, int, float, bool)):
                                span.set_attribute(
                                    f"arg.{param_name}", param_value)
                            # Serialize dict and list as JSON
                            elif isinstance(param_value, (dict, list)):
                                import json
                                try:
                                    span.set_attribute(
                                        f"arg.{param_name}", json.dumps(param_value))
                                except (TypeError, ValueError):
                                    # If not JSON serializable, just store type
                                    span.set_attribute(
                                        f"arg.{param_name}.type", type(param_value).__name__)
                            else:
                                span.set_attribute(
                                    f"arg.{param_name}.type", type(param_value).__name__)

                    # Execute async function
                    result = await func(*args, **kwargs)

                    # Calculate latency
                    latency = time.time() - start_time
                    span.set_attribute("latency_ms", latency * 1000)

                    # Capture result
                    if capture_result and result is not None:
                        if isinstance(result, (str, int, float, bool)):
                            span.set_attribute("result", result)
                        else:
                            span.set_attribute(
                                "result.type", type(result).__name__)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============================================================================
# Decorator: Azure OpenAI Specific Tracing
# ============================================================================

def trace_openai_call(
    operation: str = "chat.completions",
    capture_messages: bool = True,
    capture_response: bool = True,
):
    """
    Decorator specifically for Azure OpenAI API calls

    Args:
        operation: Operation name (e.g., "chat.completions", "embeddings")
        capture_messages: Whether to capture message content
        capture_response: Whether to capture response content

    Example:
        @trace_openai_call(operation="chat.completions")
        def chat_with_gpt(client, messages, model):
            return client.chat.completions.create(
                model=model,
                messages=messages
            )
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.start_as_current_span(
                f"azure_openai_{operation}",
                kind=SpanKind.CLIENT,
                attributes={
                    "gen_ai.system": "azure.openai",
                    "gen_ai.operation.name": operation,
                }
            ) as span:
                start_time = time.time()

                try:
                    # Try to extract common parameters
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    # Capture model
                    if 'model' in bound_args.arguments:
                        span.set_attribute(
                            "gen_ai.request.model", bound_args.arguments['model'])

                    # Capture temperature
                    if 'temperature' in bound_args.arguments:
                        span.set_attribute(
                            "gen_ai.request.temperature", bound_args.arguments['temperature'])

                    # Capture max_tokens
                    if 'max_tokens' in bound_args.arguments:
                        span.set_attribute(
                            "gen_ai.request.max_tokens", bound_args.arguments['max_tokens'])

                    # Capture messages
                    if capture_messages and 'messages' in bound_args.arguments:
                        messages = bound_args.arguments['messages']
                        if isinstance(messages, list):
                            for idx, msg in enumerate(messages):
                                if isinstance(msg, dict):
                                    span.set_attribute(
                                        f"gen_ai.prompt.{idx}.role", msg.get("role", ""))
                                    content = msg.get("content", "")
                                    if len(str(content)) < 1000:  # Limit size
                                        span.set_attribute(
                                            f"gen_ai.prompt.{idx}.content", str(content))

                    # Execute function
                    result = func(*args, **kwargs)

                    # Calculate latency
                    latency = time.time() - start_time
                    span.set_attribute(
                        "gen_ai.response.latency_ms", latency * 1000)

                    # Capture response metadata
                    if hasattr(result, 'id'):
                        span.set_attribute("gen_ai.response.id", result.id)

                    if hasattr(result, 'model'):
                        span.set_attribute(
                            "gen_ai.response.model", result.model)

                    # Capture token usage
                    if hasattr(result, 'usage') and result.usage:
                        span.set_attribute(
                            "gen_ai.usage.prompt_tokens", result.usage.prompt_tokens)
                        span.set_attribute(
                            "gen_ai.usage.completion_tokens", result.usage.completion_tokens)
                        span.set_attribute(
                            "gen_ai.usage.total_tokens", result.usage.total_tokens)

                    # Capture response content
                    if capture_response and hasattr(result, 'choices') and result.choices:
                        choice = result.choices[0]
                        if hasattr(choice, 'finish_reason'):
                            span.set_attribute(
                                "gen_ai.response.finish_reason", choice.finish_reason)
                        if hasattr(choice, 'message'):
                            span.set_attribute(
                                "gen_ai.completion.0.role", choice.message.role)
                            content = choice.message.content
                            if content and len(content) < 2000:  # Limit size
                                span.set_attribute(
                                    "gen_ai.completion.0.content", content)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.start_as_current_span(
                f"azure_openai_{operation}_async",
                kind=SpanKind.CLIENT,
                attributes={
                    "gen_ai.system": "azure.openai",
                    "gen_ai.operation.name": operation,
                }
            ) as span:
                start_time = time.time()

                try:
                    # Try to extract common parameters
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    # Capture model
                    if 'model' in bound_args.arguments:
                        span.set_attribute(
                            "gen_ai.request.model", bound_args.arguments['model'])

                    # Capture temperature
                    if 'temperature' in bound_args.arguments:
                        span.set_attribute(
                            "gen_ai.request.temperature", bound_args.arguments['temperature'])

                    # Capture max_tokens
                    if 'max_tokens' in bound_args.arguments:
                        span.set_attribute(
                            "gen_ai.request.max_tokens", bound_args.arguments['max_tokens'])

                    # Capture messages
                    if capture_messages and 'messages' in bound_args.arguments:
                        messages = bound_args.arguments['messages']
                        if isinstance(messages, list):
                            for idx, msg in enumerate(messages):
                                if isinstance(msg, dict):
                                    span.set_attribute(
                                        f"gen_ai.prompt.{idx}.role", msg.get("role", ""))
                                    content = msg.get("content", "")
                                    if len(str(content)) < 1000:
                                        span.set_attribute(
                                            f"gen_ai.prompt.{idx}.content", str(content))

                    # Execute async function
                    result = await func(*args, **kwargs)

                    # Calculate latency
                    latency = time.time() - start_time
                    span.set_attribute(
                        "gen_ai.response.latency_ms", latency * 1000)

                    # Capture response metadata
                    if hasattr(result, 'id'):
                        span.set_attribute("gen_ai.response.id", result.id)

                    if hasattr(result, 'model'):
                        span.set_attribute(
                            "gen_ai.response.model", result.model)

                    # Capture token usage
                    if hasattr(result, 'usage') and result.usage:
                        span.set_attribute(
                            "gen_ai.usage.prompt_tokens", result.usage.prompt_tokens)
                        span.set_attribute(
                            "gen_ai.usage.completion_tokens", result.usage.completion_tokens)
                        span.set_attribute(
                            "gen_ai.usage.total_tokens", result.usage.total_tokens)

                    # Capture response content
                    if capture_response and hasattr(result, 'choices') and result.choices:
                        choice = result.choices[0]
                        if hasattr(choice, 'finish_reason'):
                            span.set_attribute(
                                "gen_ai.response.finish_reason", choice.finish_reason)
                        if hasattr(choice, 'message'):
                            span.set_attribute(
                                "gen_ai.completion.0.role", choice.message.role)
                            content = choice.message.content
                            if content and len(content) < 2000:
                                span.set_attribute(
                                    "gen_ai.completion.0.content", content)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        # Return appropriate wrapper
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
