# Realtime Assistant Backend

This service exposes the backend for a voice-first Contoso support assistant. It brokers realtime audio conversations to Azure-hosted models, shares a common tool layer across providers, and exposes HTTP and WebSocket endpoints for browser or mobile clients.

## What it does

- Routes clients to Azure Voice Live.
- Provides shared tools for internal search, mock commerce flows, service booking, live-agent handoff, and optional Teams notifications.
- Loads configuration from environment variables so secrets stay out of source control.

## Project layout

- `app.py`: application entry point and route registration.
- `voice_live_handler.py`: Azure Voice Live session bridge.
- `realtime_tools.py`: shared tool result and registration classes.
- `tools.py`: shared tool implementations.
- `schemas.py`: tool schema definitions.
- `../docs/`: shared repository documentation for backend integrations and examples.

## Requirements

- Python 3.11 or newer.
- Azure Voice Live.
- Azure AI Search if you want internal search and intent lookup.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Populate `.env` with your own resource names, endpoints, and secrets.

## Run

```bash
python app.py
```

By default the server listens on `127.0.0.1:8766`. Override with `HOST` and `PORT` in `.env` if needed.

## Main configuration

| Variable | Required | Purpose |
| --- | --- | --- |
| `AZURE_VOICE_LIVE_ENDPOINT` | Yes | Azure Voice Live endpoint. |
| `AZURE_VOICE_LIVE_API_KEY` | Usually | Azure Voice Live API key. |
| `AZURE_VOICE_LIVE_MODEL` | Yes | Voice Live model or deployment name. |
| `AZURE_VOICE_LIVE_VOICE` | Optional | Default synthesized voice. |
| `AZURE_OPENAI_EMBEDDING_ENDPOINT` | Search only | Embedding endpoint for vector search. |
| `AZURE_OPENAI_EMBEDDING_API_KEY` | Search only | Embedding API key for vector search. |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Search setup only | Embedding deployment name used by indexing scripts. |
| `AZURE_OPENAI_EMBEDDING_MODEL` | Search only | Embedding model name. |
| `AZURE_SEARCH_ENDPOINT` | Search only | Azure AI Search endpoint. |
| `AZURE_SEARCH_API_KEY` | Search only | Azure AI Search key. |
| `AZURE_SEARCH_INDEX` | Search only | Knowledge base index name. |
| `TEAMS_WEBHOOK_URL` | Teams tool only | Incoming webhook or Logic App endpoint for `send_to_teams`. |
| `AZURE_TENANT_ID` | Optional | Enables Azure Developer CLI credential fallback. |

If explicit Voice Live or Search keys are not supplied, the app falls back to `AzureDeveloperCliCredential` when `AZURE_TENANT_ID` is set, otherwise `DefaultAzureCredential`.

## Endpoints

| Route | Method | Purpose |
| --- | --- | --- |
| `/test` | `GET` | Health check. |
| `/realtime` | `GET` | Voice Live websocket entrypoint. |
| `/realtime/voicelive` | `GET` | Direct Voice Live websocket. |

## Additional docs

- `../docs/EXAMPLE_PROMPTS.md`: example prompts for the Teams tool.
- `../docs/TEAMS_TOOL_README.md`: implementation notes for the Teams integration.

## Open-source hygiene

- Use `.env.example` as the only committed environment template.
- Keep local `.env` files untracked.
- Replace mock Contoso data with your own domain models before production use.
- Review tool handlers before exposing them to public traffic.
- Configure `TEAMS_WEBHOOK_URL` outside source control before using `send_to_teams`.

## Known limitations

- The Voice Live SDK is preview software.
- Mock store and booking data are demo-only.
- There is no automated test suite in this folder yet.