# Send to Teams Tool - Implementation Summary

## Overview
A new tool has been added to send messages to Microsoft Teams via Azure Logic App. The tool supports:
- Text-only messages
- Messages with images
- Messages with charts (pie, bar, line, doughnut)
- Optional title parameter

## Files Modified

### 1. `schemas.py`
Added `SEND_TO_TEAMS_SCHEMA` with the following parameters:
- **message** (required): The main text message
- **title** (optional): Title for the message card
- **image_url** (optional): Direct URL to an image
- **chart_config** (optional): Configuration for generating charts
  - chart_type: pie, bar, line, or doughnut
  - labels: Array of label strings
  - data: Array of numeric values
  - chart_title: Title for the chart
  - colors: Optional array of color codes

### 2. `tools.py`
Added two key components:

#### `_send_to_teams_tool(args)` function:
- Extracts message, title, image_url, and chart_config from arguments
- Generates chart URL using QuickChart API if chart_config is provided
- Sends POST request to Azure Logic App endpoint
- Returns success/error status with details

#### Tool registration in `attach_tools()`:
- Created `send_to_teams_target` wrapper function
- Registered in the shared tool registry
- Ensured voice_live integration with `_ensure_voice_live_tool`

## Logic App Endpoint
The tool reads its Teams/Logic App endpoint from the `TEAMS_WEBHOOK_URL` environment variable:
```
TEAMS_WEBHOOK_URL=https://your-logic-app-or-teams-webhook
```

For a reusable Azure Logic App example, see `../logic-app/README.md` and `../logic-app/teams-http-notifier.logicapp.json`.

## Chart Generation
Charts are generated using QuickChart.io API:
- Supports pie, bar, line, and doughnut charts
- Default chart size: 1000x700 pixels
- White background
- Default color palette: ['#4CAF50', '#2196F3', '#FFC107', '#E91E63', '#9C27B0', '#FF5722']
- URL encoding for JSON configuration

## Usage Examples

### Example 1: Text-only message
```python
args = {
    "message": "Hello from AI Assistant!",
    "title": "Notification"
}
```

### Example 2: Message with image
```python
args = {
    "message": "Check out this image!",
    "title": "Photo",
    "image_url": "https://example.com/image.jpg"
}
```

### Example 3: Message with pie chart
```python
args = {
    "message": "Q4 Revenue Distribution",
    "title": "Revenue Report",
    "chart_config": {
        "chart_type": "pie",
        "labels": ["Sales", "Marketing", "Engineering"],
        "data": [350, 180, 220],
        "chart_title": "Revenue by Department"
    }
}
```

### Example 4: Message with bar chart
```python
args = {
    "message": "Monthly active users",
    "title": "User Growth",
    "chart_config": {
        "chart_type": "bar",
        "labels": ["Jan", "Feb", "Mar"],
        "data": [1200, 1900, 1500]
    }
}
```

## Testing
A test script has been created: `test_teams_tool.py`

Run all tests:
```bash
python test_teams_tool.py
```

Run specific test:
```bash
python test_teams_tool.py 1    # Text-only
python test_teams_tool.py 4    # Pie chart
python test_teams_tool.py 5    # Bar chart
```

## Response Format
The tool returns a JSON response:

### Success:
```json
{
    "status": "success",
    "message": "Message sent to Teams successfully",
    "title": "Message Title",
    "has_image": true,
    "has_chart": false
}
```

### Error:
```json
{
    "status": "error",
    "error_code": 500,
    "error_message": "Error details"
}
```

## Integration with Voice Assistant
The tool is automatically available in:
- Voice live handler (if configured)

The AI assistant can now call this tool when users request to:
- Send notifications to Teams
- Share reports or analytics
- Post visualizations
- Send status updates

## Dependencies
- `aiohttp`: For async HTTP requests
- `urllib.parse.quote`: For URL encoding
- QuickChart.io API (external service, no authentication required)

## Notes
- All requests are asynchronous using `aiohttp`
- Chart URLs are generated dynamically based on configuration
- The Logic App expects JSON payload with: message, title (optional), imageUrl (optional)
- HTTP 202 status indicates successful message queuing in Logic App