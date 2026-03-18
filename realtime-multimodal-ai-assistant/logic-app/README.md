# Logic App Sample

This folder contains an Azure Logic App example that the realtime backend can call through the `TEAMS_WEBHOOK_URL` setting.

## Files

- `teams-http-notifier.logicapp.json`: Sanitized Logic App template based on the exported workflow.
- `sample-request.json`: Example HTTP payload for local testing.

## What This Logic App Does

The workflow exposes an HTTP trigger and then:

- posts a plain Teams message when only `message` is provided
- posts an Adaptive Card when `imageUrl` is also provided
- uses `title` when present and falls back to `Message`

## Before You Use It

Replace the placeholder values in `teams-http-notifier.logicapp.json`:

- `alerts@contoso.com` with the target Teams user or chat recipient you want to notify
- `<subscription-id>` with your Azure subscription ID
- `<resource-group>` with the resource group that owns the Teams API connection
- `<location>` with the Azure region of the managed API and connection

## Create The Logic App In Azure Portal

1. Create a new Azure Logic App.
2. Add an HTTP request trigger if you are building manually, or open Code View if you want to paste the workflow definition directly.
3. Copy the workflow from `teams-http-notifier.logicapp.json`.
4. Create or authorize a Microsoft Teams API connection named `teams`.
5. Save the workflow so Azure generates the HTTP POST URL.
6. Copy the generated trigger URL into the backend `.env` file as `TEAMS_WEBHOOK_URL`.

## Expected Request Body

```json
{
  "message": "Quarterly status update is ready",
  "title": "Status Update",
  "imageUrl": "https://example.com/chart.png"
}
```

Only `message` is required.

## Example Test

```bash
curl -X POST "$TEAMS_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d @sample-request.json
```

## Backend Integration

The realtime backend reads the webhook from:

```env
TEAMS_WEBHOOK_URL=https://your-logic-app-trigger-url
```

See `../docs/TEAMS_TOOL_README.md` for the tool-level integration details.