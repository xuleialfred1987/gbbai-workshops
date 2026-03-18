import logging
import os
import re
import json
import random
import string
from typing import Any, Optional, Callable, Awaitable

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from openai import AzureOpenAI

try:
    from azure.search.documents.models import VectorizedQuery
except ImportError:
    from azure.search.documents._generated.models import VectorizedQuery

from realtime_tools import Tool, ToolResult, ToolResultDirection
from schemas import (
    ADD_TO_CART_SCHEMA,
    BOOK_CS_CENTER_SCHEMA,
    # IMAGE_CAPTION_SCHEMA,
    INTENT_SEARCH_SCHEMA,
    PHONE_STORE_SEARCH_SCHEMA,
    INTERNAL_SEARCH_SCHEMA,
    REPORT_GROUNDING_SCHEMA,
    SEND_TO_TEAMS_SCHEMA,
    TRANSFER_TO_LIVE_AGENT_SCHEMA,
)


logger = logging.getLogger("TOOLS")


def _create_embedding_client_from_env() -> tuple[Optional[AzureOpenAI], Optional[str], Optional[int]]:
    aoai_endpoint = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT")
    aoai_key = os.environ.get("AZURE_OPENAI_EMBEDDING_API_KEY")
    aoai_api_version = os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION") or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
    embed_model = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL")
    embed_dimensions = os.environ.get("AZURE_OPENAI_EMBEDDING_DIMENSIONS")

    if not aoai_endpoint or not aoai_key or not embed_model:
        return None, None, None

    client = AzureOpenAI(
        api_key=aoai_key,
        azure_endpoint=aoai_endpoint,
        api_version=aoai_api_version,
    )
    dimensions = int(embed_dimensions) if embed_dimensions else None
    return client, embed_model, dimensions


async def _embed_query(
        embedding_client: Optional[AzureOpenAI],
        embedding_model: Optional[str],
        embedding_dimensions: Optional[int],
        text: str) -> list[float]:
    if not embedding_client or not embedding_model:
        raise RuntimeError(
            "Azure OpenAI embeddings not configured. Set AZURE_OPENAI_EMBEDDING_ENDPOINT, AZURE_OPENAI_EMBEDDING_API_KEY, and AZURE_OPENAI_EMBEDDING_MODEL."
        )

    kwargs: dict[str, Any] = {
        "model": embedding_model,
        "input": text,
    }
    if embedding_dimensions is not None:
        kwargs["dimensions"] = embedding_dimensions

    response = embedding_client.embeddings.create(**kwargs)
    return response.data[0].embedding


async def _image_caption(args: Any) -> ToolResult:

    output = {
        'timestamp': 1744792132,
        'caption': """The image features a person sitting at a desk, wearing a light blue t-shirt. They are using a computer or laptop, indicated by their hands positioned near the keyboard. The background includes a well-organized room with light-colored walls.

To the left, there is a clothing rack displaying various garments, suggesting a tidy and personal space. The rack is filled with a mix of clothes, including what appears to be dresses and shirts.

In the right background, there is a neatly made bed with a soft, patterned blanket and a gray item, possibly a piece of clothing or fabric, lying on it. The floor is wooden, and there is a white office chair with a mesh back, positioned in front of the desk. The overall atmosphere of the room conveys a sense of order and casual comfort."""
    }

    result = json.dumps(output)

    return ToolResult(result, ToolResultDirection.TO_SERVER)


async def _book_cs_center_tool(args: Any) -> ToolResult:
    # Extract parameters
    customer_name = args["customer_name"]
    phone_number = args["phone_number"]
    device_model = args["device_model"]
    preferred_date = args["preferred_date"]
    preferred_time = args["preferred_time"]
    service_type = args["service_type"]
    preferred_location = args["preferred_location"]

    device_images = {
        "Galaxy S25 Ultra": "/assets/images/smart-phone/galaxy-s25-ultra.jpeg",
        "Galaxy S25+": "/assets/images/smart-phone/galaxy-s25-plus.jpeg",
        "Galaxy S25 Plus": "/assets/images/smart-phone/galaxy-s25-plus.jpeg",
        "Galaxy Z Fold6": "/assets/images/smart-phone/galaxy-z-fold6.jpeg",
        "Galaxy S25": "/assets/images/smart-phone/galaxy-s25.jpeg",
    }

    # Generate a booking id
    booking_id = "bk-" + \
        ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))

    # Mock booking confirmation (replace with real booking logic as needed)
    confirmation = {
        "status": "success",
        "booking_id": booking_id,
        "message": (
            f"Appointment booked for {customer_name} ({phone_number}) at {preferred_location}.\n"
            f"Device: {device_model}\n"
            f"Service: {service_type}\n"
            f"Date: {preferred_date} Time: {preferred_time}"
        ),
        "customer": customer_name,
        "phone": phone_number,
        "device": device_model,
        "device_image": device_images.get(device_model, ""),
        "date": preferred_date,
        "time": preferred_time,
        "service": service_type,
        "location": preferred_location
    }

    return ToolResult(json.dumps(confirmation), ToolResultDirection.TO_SERVER)


async def _search_phone_store_tool(args: Any) -> ToolResult:
    phone_model = args['phone_model'].lower()

    # Mock store information
    # titles = [
    #     'Cell Phone Accessories',
    #     'Tech Pod',
    #     'Smart Phone Store',
    #     'Cell Spot',
    #     'Gadget Galaxy',
    #     'Digi Mart',
    #     'Phone Zone',
    # ]

    titles = [
        'Contoso Phone Ultra',
        'Contoso Phone Plus',
        'Contoso Fold',
        'Contoso Phone',
    ]

    images = [
        '/assets/images/smart-phone/galaxy-s25-ultra.jpeg',
        '/assets/images/smart-phone/galaxy-s25-plus.jpeg',
        '/assets/images/smart-phone/galaxy-z-fold6.jpeg',
        '/assets/images/smart-phone/galaxy-s25.jpeg',
    ]

    # Generate mock store data
    store_data = []
    for i in range(4):
        store_data.append({
            'title': titles[i],
            'address': 'shop.contoso.com',
            'distance': f"{random.randint(200, 800)}m",
            'rating': round(random.uniform(3.0, 5.0), 1),
            'reviews': random.randint(100, 5000),
            'perCapitaConsumption': random.randint(100, 500),
            'image': images[i],
        })

    # Mock response - in a real implementation, this would query a database or API
    selected_stores = []
    if "contoso phone ultra" in phone_model:
        selected_stores = random.sample(store_data, min(5, len(store_data)))
    elif "contoso fold" in phone_model:
        selected_stores = random.sample(store_data, min(5, len(store_data)))
    else:
        selected_stores = random.sample(store_data, min(5, len(store_data)))

    return ToolResult(json.dumps(selected_stores), ToolResultDirection.TO_SERVER)


async def _add_to_cart_tool(args: Any) -> ToolResult:
    phone_model = args['phone_model'].lower()

    data = {
        "product": phone_model,
        "status": "Added to cart successfully",
        "location": "Contoso Flagship Store",
        "price": random.randint(1000, 2500),
        "currency": "USD",
        "availability": "In Stock",
        "estimated_delivery": "2-3 business days",
        "warranty": "1 year manufacturer warranty"
    }

    return ToolResult(json.dumps(data), ToolResultDirection.TO_SERVER)


async def _send_to_teams_tool(args: Any) -> ToolResult:
    """Send a message to Microsoft Teams via a configured webhook endpoint."""
    import aiohttp
    from urllib.parse import quote

    logic_app_url = os.environ.get("TEAMS_WEBHOOK_URL")

    message = args.get("message")
    title = args.get("title")
    image_url = args.get("image_url")
    chart_config = args.get("chart_config")

    if not logic_app_url:
        result = {
            "status": "error",
            "error_message": "TEAMS_WEBHOOK_URL is not configured."
        }
        wrapped_payload = {
            "tool": "send_to_teams",
            "result": result,
            "original_message": message,
            "title": title,
            "imageUrl": image_url
        }
        return ToolResult(json.dumps(wrapped_payload), ToolResultDirection.TO_SERVER)

    # Generate chart URL if chart_config is provided
    if chart_config and not image_url:
        chart_type = chart_config.get("chart_type")
        labels = chart_config.get("labels", [])
        data = chart_config.get("data", [])
        chart_title = chart_config.get("chart_title", "Chart")
        colors = chart_config.get("colors", [
            '#4CAF50', '#2196F3', '#FFC107', '#E91E63', '#9C27B0', '#FF5722'
        ])

        chart_definition = {
            'type': chart_type,
            'data': {
                'labels': labels,
                'datasets': [{
                    'data': data,
                    'backgroundColor': colors[:len(data)]
                }]
            }
        }

        if chart_title:
            chart_definition['options'] = {
                'title': {'display': True, 'text': chart_title}}

        image_url = f"https://quickchart.io/chart?width=1000&height=700&backgroundColor=white&c={quote(json.dumps(chart_definition))}"

    # Prepare payload
    payload = {"message": message}
    if title:
        payload["title"] = title
    if image_url:
        payload["imageUrl"] = image_url

    # Send request to Logic App
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                logic_app_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                status = response.status
                response_text = await response.text()

                if status == 202:
                    result = {
                        "status": "success",
                        "message": "Message sent to Teams successfully",
                        "title": title or "Message",
                        "has_image": bool(image_url),
                        "has_chart": bool(chart_config)
                    }
                else:
                    result = {
                        "status": "error",
                        "error_code": status,
                        "error_message": response_text
                    }

    except Exception as e:
        result = {
            "status": "error",
            "error_message": str(e)
        }

    # Wrap the result in a structured payload
    wrapped_payload = {
        "tool": "send_to_teams",
        "result": result,
        "original_message": message,
        "title": title,
        "imageUrl": image_url
    }

    return ToolResult(json.dumps(wrapped_payload), ToolResultDirection.TO_SERVER)


def _normalize_transcript_entries(entries: Any) -> list[dict[str, Any]]:
    normalized_entries: list[dict[str, Any]] = []

    if not isinstance(entries, list):
        return normalized_entries

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        role = str(entry.get("role") or "").strip().lower()
        text = str(entry.get("text") or "").strip()
        created_at = entry.get("created_at")

        if role not in {"user", "assistant"} or not text:
            continue

        normalized_entries.append(
            {
                "role": role,
                "text": text,
                "created_at": created_at,
            }
        )

    return normalized_entries


def _format_transcript(entries: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    for entry in entries:
        speaker = "User" if entry["role"] == "user" else "AI Assistant"
        lines.append(f"{speaker}: {entry['text']}")

    return "\n".join(lines)


async def _transfer_to_live_agent_tool(args: Any) -> ToolResult:
    import aiohttp

    session_context = args.get("_session_context") or {}
    transcript_entries = _normalize_transcript_entries(
        session_context.get("conversation_transcript") or args.get("conversation_transcript")
    )
    transcript_text = _format_transcript(transcript_entries)
    reason = str(args.get("reason") or "User requested live agent assistance.").strip()
    issue_summary = str(args.get("issue_summary") or "Support handoff requested.").strip()
    intent_key = str(args.get("intent_key") or "").strip() or None
    serial_number = str(args.get("serial_number") or "").strip() or None
    handoff_id = "la-" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    handoff_url = os.environ.get("LIVE_AGENT_HANDOFF_URL")
    handoff_api_key = os.environ.get("LIVE_AGENT_HANDOFF_API_KEY")

    handoff_payload: dict[str, Any] = {
        "handoff_id": handoff_id,
        "status": "success",
        "destination": os.environ.get("LIVE_AGENT_HANDOFF_DESTINATION") or "Contoso live agent",
        "reason": reason,
        "issue_summary": issue_summary,
        "intent_key": intent_key,
        "serial_number": serial_number,
        "transcript": transcript_entries,
        "transcript_text": transcript_text,
        "transcript_line_count": len(transcript_entries),
        "delivery": {
            "mode": "local",
            "configured_endpoint": bool(handoff_url),
        },
    }

    if handoff_url:
        headers = {"Content-Type": "application/json"}
        if handoff_api_key:
            headers["x-api-key"] = handoff_api_key

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(handoff_url, json=handoff_payload, headers=headers) as response:
                    response_text = await response.text()
                    if 200 <= response.status < 300:
                        handoff_payload["delivery"] = {
                            "mode": "remote",
                            "configured_endpoint": True,
                            "status_code": response.status,
                            "response_preview": response_text[:300],
                        }
                    else:
                        handoff_payload["status"] = "error"
                        handoff_payload["error_message"] = (
                            f"Live agent handoff endpoint returned {response.status}: {response_text[:300]}"
                        )
                        handoff_payload["delivery"] = {
                            "mode": "remote",
                            "configured_endpoint": True,
                            "status_code": response.status,
                            "response_preview": response_text[:300],
                        }
        except Exception as exc:
            handoff_payload["status"] = "error"
            handoff_payload["error_message"] = str(exc)
            handoff_payload["delivery"] = {
                "mode": "remote",
                "configured_endpoint": True,
            }
    else:
        handoff_payload["delivery"]["message"] = (
            "No LIVE_AGENT_HANDOFF_URL configured. A local handoff package was generated for the UI only."
        )

    wrapped_payload = {
        "tool": "transfer_to_live_agent",
        "result": handoff_payload,
    }

    return ToolResult(json.dumps(wrapped_payload, ensure_ascii=False), ToolResultDirection.TO_SERVER)


async def _report_grounding_tool(args: Any) -> ToolResult:
    sources = args.get("sources") or []
    logger.info("report_grounding called with sources=%s", sources)
    payload = {
        "status": "ok",
        "sources_recorded": sources,
        "count": len(sources),
    }
    return ToolResult(json.dumps(payload, ensure_ascii=False), ToolResultDirection.TO_SERVER)


def _ensure_voice_live_tool(
        voice_live_mt: Optional[Any],
        tool_schema: dict[str, Any],
        tool_target: Callable[[Any], Awaitable[ToolResult]]
) -> None:
    if not voice_live_mt:
        return

    existing_tools = getattr(voice_live_mt, "tools", [])
    if any(existing_tool.get("name") == tool_schema.get("name") for existing_tool in existing_tools):
        # Even if schema already present, ensure handler is up to date
        voice_live_mt.add_tool(tool_schema, tool_target)
        return

    voice_live_mt.add_tool(tool_schema, tool_target)


async def _search_tool(
        search_client: SearchClient,
        semantic_configuration: str,
        identifier_field: str,
        content_field: str,
        embedding_field: str,
    language_field: Optional[str],
        title_field: Optional[str],
        use_vector_query: bool,
    embedding_client: Optional[AzureOpenAI],
    embedding_model: Optional[str],
    embedding_dimensions: Optional[int],
        args: Any) -> ToolResult:
    query = (args.get("query") or "").strip()

    if not query:
        payload = json.dumps({
            "error": "Query must be provided.",
            "chunks": [],
            "count": 0,
        })
        return ToolResult(payload, ToolResultDirection.TO_CLIENT)

    logger.info(
        "internal_search called: query=%s semantic_config=%s use_vector=%s index_fields={id:%s,title:%s,content:%s,embedding:%s,language:%s}",
        query,
        semantic_configuration,
        use_vector_query,
        identifier_field,
        title_field,
        content_field,
        embedding_field,
        language_field,
    )
    print(f"Searching for '{query}' in the knowledge base.")

    # Hybrid + reranking query using Azure AI Search
    vector_queries = []
    if use_vector_query:
        try:
            embedded_query = await _embed_query(
                embedding_client,
                embedding_model,
                embedding_dimensions,
                query,
            )
            vector_queries.append(VectorizedQuery(
                vector=embedded_query,
                k_nearest_neighbors=50,
                fields=embedding_field,
            ))
        except Exception as exc:
            logger.warning(
                "Falling back to Search-side vectorization for internal_search: %s",
                exc,
            )
            vector_queries.append(VectorizableTextQuery(
                text=query, k_nearest_neighbors=50, fields=embedding_field))

    select_fields = [identifier_field, content_field]
    if title_field and title_field not in select_fields:
        select_fields.append(title_field)
    if language_field and language_field not in select_fields:
        select_fields.append(language_field)

    search_results = await search_client.search(
        search_text=query,
        query_type="semantic",
        semantic_configuration_name=semantic_configuration,
        top=3,
        vector_queries=vector_queries or None,
        select=", ".join(select_fields)
    )

    chunks: list[dict[str, Any]] = []
    formatted_sections: list[str] = []

    async for result in search_results:
        chunk_id = result.get(identifier_field)
        chunk_content = result.get(content_field)
        chunk: dict[str, Any] = {
            "id": chunk_id,
            "content": chunk_content,
            "score": result.get("@search.score"),
            "reranker_score": result.get("@search.reranker_score"),
        }
        if title_field:
            chunk["title"] = result.get(title_field)
        if language_field:
            chunk["language"] = result.get(language_field)

        chunks.append(chunk)
        formatted_sections.append(
            f"[{chunk_id}]: {chunk_content}\n-----\n"
        )
    if chunks:
        logger.info(
            "internal_search returned %s chunk(s); first_chunk=%s",
            len(chunks),
            chunks[0],
        )
    else:
        logger.info("internal_search returned 0 chunks")

    results_text = "".join(formatted_sections).strip()

    payload: dict[str, Any] = {
        "query": query,
        "answer_language_rule": "Respond in the user's current language. Do not switch languages because the knowledge-base content is in another language; translate or paraphrase the source when needed.",
        "results_text": results_text,
        "chunks": chunks,
        "count": len(chunks),
    }

    if not chunks:
        payload["message"] = "No results found."

    return ToolResult(json.dumps(payload, ensure_ascii=False), ToolResultDirection.TO_SERVER)


async def _intent_search_tool(
        search_client: SearchClient,
        identifier_field: str,
    text_field: Optional[str],
        intent_key_field: str,
        embedding_field: str,
        use_vector_query: bool,
        k_nearest_neighbors: int,
        score_threshold: float,
    embedding_client: Optional[AzureOpenAI],
    embedding_model: Optional[str],
    embedding_dimensions: Optional[int],
        args: Any) -> ToolResult:
    query = (args.get("query") or "").strip()

    if not query:
        payload = json.dumps({
            "error": "Query must be provided.",
            "count": 0,
        })
        return ToolResult(payload, ToolResultDirection.TO_SERVER)

    logger.info(
        "intent_search called: query=%s use_vector=%s text_field=%s embedding_field=%s threshold=%s",
        query,
        use_vector_query,
        text_field,
        embedding_field,
        score_threshold,
    )

    vector_queries = None
    if use_vector_query:
        try:
            embedded_query = await _embed_query(
                embedding_client,
                embedding_model,
                embedding_dimensions,
                query,
            )
            vector_queries = [
                VectorizedQuery(
                    vector=embedded_query,
                    k_nearest_neighbors=k_nearest_neighbors,
                    fields=embedding_field,
                )
            ]
        except Exception as exc:
            logger.warning(
                "Falling back to Search-side vectorization for intent_search: %s",
                exc,
            )
            vector_queries = [
                VectorizableTextQuery(
                    text=query,
                    k_nearest_neighbors=k_nearest_neighbors,
                    fields=embedding_field,
                )
            ]

    select_fields = [identifier_field, intent_key_field]
    if text_field and text_field not in select_fields:
        select_fields.append(text_field)

    try:
        results = await search_client.search(
            search_text=query,
            query_type="simple",
            top=1,
            vector_queries=vector_queries,
            select=", ".join(select_fields),
        )
    except Exception as exc:
        logger.warning("Intent search with vector query failed, retrying without vector query: %s", exc)
        results = await search_client.search(
            search_text=query,
            query_type="simple",
            top=1,
            select=", ".join(select_fields),
        )

    best_result = None
    async for result in results:
        best_result = result
        break

    score = best_result.get("@search.score") if best_result else None
    if best_result and score is not None and score < score_threshold:
        best_result = None

    payload = {
        "query": query,
        "count": 1 if best_result else 0,
        "id": best_result.get(identifier_field) if best_result else None,
        "text": best_result.get(text_field) if best_result and text_field else None,
        "intent_key": best_result.get(intent_key_field) if best_result else None,
        "score": score,
    }

    logger.info("intent_search result: %s", payload)

    return ToolResult(json.dumps(payload, ensure_ascii=False), ToolResultDirection.TO_SERVER)

KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_=\-]+$')

# TODO: move from sending all chunks used for grounding eagerly to only sending links to
# the original content in storage, it'll be more efficient overall


def attach_tools(credentials: AzureKeyCredential | DefaultAzureCredential,
                 search_endpoint: str, search_index: str,
                 semantic_configuration: str,
                 identifier_field: str,
                 content_field: str,
                 embedding_field: str,
                 title_field: str,
                 use_vector_query: bool,
                 tool_registry: Optional[dict[str, Tool]] = None,
                 voice_live_mt: Optional[Any] = None
                 ) -> None:
    tool_registry = tool_registry if tool_registry is not None else {}
    search_client: Optional[SearchClient] = None
    intent_search_client: Optional[SearchClient] = None
    search_configured = bool(search_endpoint and search_index)
    search_language_field = os.environ.get("AZURE_SEARCH_LANGUAGE_FIELD")
    intent_search_endpoint = os.environ.get("AZURE_INTENT_SEARCH_ENDPOINT") or search_endpoint
    intent_search_index = os.environ.get("AZURE_INTENT_SEARCH_INDEX")
    intent_identifier_field = os.environ.get("AZURE_INTENT_IDENTIFIER_FIELD") or "id"
    intent_text_field = os.environ.get("AZURE_INTENT_TEXT_FIELD") or "historical_question"
    intent_embedding_field = os.environ.get("AZURE_INTENT_EMBEDDING_FIELD") or "historical_question_vec"
    intent_key_field = os.environ.get("AZURE_INTENT_INTENT_KEY_FIELD") or "intent_key"
    intent_use_vector_query = os.environ.get("AZURE_INTENT_USE_VECTOR", "true").lower() == "true"
    intent_k_nearest_neighbors = int(os.environ.get("AZURE_INTENT_K_NEAREST_NEIGHBORS", "10"))
    intent_score_threshold = float(os.environ.get("AZURE_INTENT_SCORE_THRESHOLD", "0"))
    embedding_client, embedding_model, embedding_dimensions = _create_embedding_client_from_env()

    logger.info(
        "attach_tools config: search_endpoint=%s search_index=%s intent_endpoint=%s intent_index=%s use_vector=%s intent_use_vector=%s embedding_model=%s",
        search_endpoint,
        search_index,
        intent_search_endpoint,
        intent_search_index,
        use_vector_query,
        intent_use_vector_query,
        embedding_model,
    )

    if search_configured:
        if credentials is None:
            logger.warning(
                "Azure Search credentials not supplied; internal_search tool will return errors.")
        else:
            if not isinstance(credentials, AzureKeyCredential):
                credentials.get_token("https://search.azure.com/.default")
            search_client = SearchClient(
                search_endpoint,
                search_index,
                credentials,
                user_agent="VoiceLiveMiddleTier"
            )

    if intent_search_endpoint and intent_search_index:
        if credentials is None:
            logger.warning(
                "Azure Search credentials not supplied; intent_search tool will return errors.")
        else:
            intent_search_client = SearchClient(
                intent_search_endpoint,
                intent_search_index,
                credentials,
                user_agent="VoiceLiveMiddleTier"
            )

    if search_client is None:
        if search_configured:
            logger.warning(
                "Azure Search client could not be created; internal_search tool will return errors.")
        else:
            logger.warning(
                "Azure Search endpoint/index not configured. internal_search tool will return errors.")

        async def internal_search_target(args: Any) -> ToolResult:
            message = json.dumps({
                "error": "Azure Search configuration is missing.",
                "details": "Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX to enable internal_search."
            })
            return ToolResult(message, ToolResultDirection.TO_CLIENT)
    else:
        async def internal_search_target(args: Any) -> ToolResult:
            return await _search_tool(
                search_client, semantic_configuration, identifier_field,
                content_field, embedding_field, search_language_field,
                title_field, use_vector_query, embedding_client,
                embedding_model, embedding_dimensions, args)

    tool_registry["internal_search"] = Tool(
        schema=INTERNAL_SEARCH_SCHEMA,
        target=internal_search_target
    )
    logger.info("Registered tool: internal_search")
    _ensure_voice_live_tool(
        voice_live_mt, INTERNAL_SEARCH_SCHEMA, internal_search_target)

    if intent_search_client is None:
        async def intent_search_target(args: Any) -> ToolResult:
            message = json.dumps({
                "error": "Intent search configuration is missing.",
                "details": "Set AZURE_INTENT_SEARCH_INDEX to enable intent_search."
            })
            return ToolResult(message, ToolResultDirection.TO_CLIENT)
    else:
        async def intent_search_target(args: Any) -> ToolResult:
            return await _intent_search_tool(
                intent_search_client,
                intent_identifier_field,
                intent_text_field,
                intent_key_field,
                intent_embedding_field,
                intent_use_vector_query,
                intent_k_nearest_neighbors,
                intent_score_threshold,
                embedding_client,
                embedding_model,
                embedding_dimensions,
                args,
            )

    tool_registry["intent_search"] = Tool(
        schema=INTENT_SEARCH_SCHEMA,
        target=intent_search_target
    )
    logger.info("Registered tool: intent_search")
    _ensure_voice_live_tool(
        voice_live_mt, INTENT_SEARCH_SCHEMA, intent_search_target)

    async def report_grounding_target(args: Any) -> ToolResult:
        return await _report_grounding_tool(args)
    tool_registry["report_grounding"] = Tool(
        schema=REPORT_GROUNDING_SCHEMA,
        target=report_grounding_target
    )
    logger.info("Registered tool: report_grounding")
    _ensure_voice_live_tool(
        voice_live_mt, REPORT_GROUNDING_SCHEMA, report_grounding_target)

    async def add_to_cart_target(args: Any) -> ToolResult:
        return await _add_to_cart_tool(args)
    tool_registry["add_to_cart"] = Tool(
        schema=ADD_TO_CART_SCHEMA,
        target=add_to_cart_target
    )
    logger.info("Registered tool: add_to_cart")
    _ensure_voice_live_tool(
        voice_live_mt, ADD_TO_CART_SCHEMA, add_to_cart_target)

    async def search_phone_store_target(args: Any) -> ToolResult:
        return await _search_phone_store_tool(args)
    tool_registry["search_phone_store"] = Tool(
        schema=PHONE_STORE_SEARCH_SCHEMA,
        target=search_phone_store_target
    )
    logger.info("Registered tool: search_phone_store")
    _ensure_voice_live_tool(
        voice_live_mt, PHONE_STORE_SEARCH_SCHEMA, search_phone_store_target)

    async def book_cs_center_target(args: Any) -> ToolResult:
        return await _book_cs_center_tool(args)
    tool_registry["book_cs_center"] = Tool(
        schema=BOOK_CS_CENTER_SCHEMA,
        target=book_cs_center_target
    )
    logger.info("Registered tool: book_cs_center")
    _ensure_voice_live_tool(
        voice_live_mt, BOOK_CS_CENTER_SCHEMA, book_cs_center_target)

    async def send_to_teams_target(args: Any) -> ToolResult:
        return await _send_to_teams_tool(args)
    tool_registry["send_to_teams"] = Tool(
        schema=SEND_TO_TEAMS_SCHEMA,
        target=send_to_teams_target
    )
    logger.info("Registered tool: send_to_teams")
    _ensure_voice_live_tool(
        voice_live_mt, SEND_TO_TEAMS_SCHEMA, send_to_teams_target)

    async def transfer_to_live_agent_target(args: Any) -> ToolResult:
        return await _transfer_to_live_agent_tool(args)
    tool_registry["transfer_to_live_agent"] = Tool(
        schema=TRANSFER_TO_LIVE_AGENT_SCHEMA,
        target=transfer_to_live_agent_target
    )
    logger.info("Registered tool: transfer_to_live_agent")
    _ensure_voice_live_tool(
        voice_live_mt, TRANSFER_TO_LIVE_AGENT_SCHEMA, transfer_to_live_agent_target)

    # tool_registry["image_caption"] = Tool(
    #     schema=IMAGE_CAPTION_SCHEMA,
    #     target=lambda args: _image_caption(args)
    # )
