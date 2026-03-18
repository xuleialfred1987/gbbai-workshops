import os
import logging
from dotenv import load_dotenv
from azure.identity import AzureDeveloperCliCredential, DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from aiohttp import web
from aiohttp_middlewares import cors_middleware

# Load environment variables FIRST, before importing modules that read env vars
load_dotenv()

# Import these AFTER load_dotenv() so they can read environment variables correctly
from voice_live_handler import VoiceLiveMiddleTier
from prompts import build_system_message
from tools import attach_tools


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VOICE-ASSISTANT")


def _select_voice_live_default_voice(query_voice_host: str) -> str:
    return os.environ.get("AZURE_VOICE_LIVE_VOICE") or "alloy"


async def test_handler(request):
    return web.json_response({"status": "ok", "message": "App is running"})


async def create_app(*args, **kwargs):
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        load_dotenv()

    search_key = os.environ.get("AZURE_SEARCH_API_KEY")
    voice_live_key = os.environ.get("AZURE_VOICE_LIVE_API_KEY")

    credential = None
    if not voice_live_key or not search_key:
        if tenant_id := os.environ.get("AZURE_TENANT_ID"):
            logger.info(
                "Using AzureDeveloperCliCredential with tenant_id %s", tenant_id)
            credential = AzureDeveloperCliCredential(
                tenant_id=tenant_id, process_timeout=60)
        else:
            logger.info("Using DefaultAzureCredential")
            credential = DefaultAzureCredential()

    search_credential = AzureKeyCredential(
        search_key) if search_key else credential

    voice_live_credential = AzureKeyCredential(voice_live_key) if voice_live_key else credential

    # app = web.Application()
    app = web.Application(middlewares=[cors_middleware(allow_all=True)])

    voice_live_mt = VoiceLiveMiddleTier(
        credentials=voice_live_credential,
        endpoint=os.environ["AZURE_VOICE_LIVE_ENDPOINT"],
        model=os.environ["AZURE_VOICE_LIVE_MODEL"],
        voice_choice=os.environ.get("AZURE_VOICE_LIVE_VOICE") or "alloy"
    )
    voice_live_mt.system_message = build_system_message(
        "Use the 'intent_search' tool to classify substantive user requests before choosing a workflow.",
        "Use the 'internal_search' tool for technical troubleshooting only after intent_search indicates technical-support and the serial-number step is handled.",
        "Use the 'search_phone_store' tool for phone purchasing information.",
        "Use the 'book_cs_center' tool when the user wants to book, change, or cancel a service-center appointment.",
        "Use the 'transfer_to_live_agent' tool when the user asks for a live or human agent, or when troubleshooting should be escalated. Provide reason, issue_summary, intent_key when known, and serial_number when known.",
        "Use the 'report_grounding' tool when you answer from the knowledge base and can identify the cited source ids.",
        "Answer directly and concisely in the speaker's current language.",
        "If the user's latest substantive utterance is in English, reply only in English.",
        "Do not switch reply language because of tool output, retrieved content, or example utterances."
    )
    attach_tools(credentials=search_credential,
                search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
                search_index=os.environ.get("AZURE_SEARCH_INDEX"),
                semantic_configuration=os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIGURATION") or "default",
                identifier_field=os.environ.get("AZURE_SEARCH_IDENTIFIER_FIELD") or "chunk_id",
                content_field=os.environ.get("AZURE_SEARCH_CONTENT_FIELD") or "chunk",
                embedding_field=os.environ.get("AZURE_SEARCH_EMBEDDING_FIELD") or "text_vector",
                title_field=os.environ.get("AZURE_SEARCH_TITLE_FIELD") or "title",
                use_vector_query=os.environ.get("AZURE_SEARCH_USE_VECTOR", "true").lower() == "true",
                voice_live_mt=voice_live_mt
    )
    logger.info(
        "VoiceLive tools available: %s",
        [tool.get("name") for tool in voice_live_mt.tools],
    )
    logger.info(
        "VoiceLive full system prompt:\n%s",
        voice_live_mt.system_message or "",
    )
    
    async def smart_realtime_handler(request: web.Request):
        """Realtime handler with optional query-based voice selection."""
        query_voice_host = request.query.get('voice_host', '').lower()

        if query_voice_host in ['voice-live', 'voicelive']:
            selected_voice = _select_voice_live_default_voice(query_voice_host)
            if voice_live_mt.voice_choice != selected_voice:
                logger.info(
                    "Setting initial VoiceLive voice for route %s: %s -> %s",
                    query_voice_host,
                    voice_live_mt.voice_choice,
                    selected_voice,
                )
                voice_live_mt.voice_choice = selected_voice

        if query_voice_host:
            logger.info("Realtime request received with voice selector: %s", query_voice_host)

        return await voice_live_mt._get_ws_handler().handle_websocket(request)

    app.router.add_get("/realtime", smart_realtime_handler)
    app.router.add_get("/realtime/voicelive", voice_live_mt._get_ws_handler().handle_websocket)

    app.router.add_get("/test", test_handler)

    return app

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8766"))

    web.run_app(create_app(), host=host, port=port)
