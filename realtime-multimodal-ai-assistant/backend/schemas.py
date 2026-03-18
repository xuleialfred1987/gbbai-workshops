"""
This file contains schema definitions for tools used in the application.
Centralizing schemas makes them easier to manage and update.
"""


TRANSFER_TO_LIVE_AGENT_SCHEMA = {
    "type": "function",
    "name": "transfer_to_live_agent",
    "description": (
        "Transfer the current case to a live Contoso support agent. "
        "Use this when the user explicitly asks for a live or human agent, "
        "when troubleshooting is exhausted, or when the case should be escalated. "
        "The server automatically attaches the recent conversation transcript, so do not paste it into the arguments."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why the case needs to be transferred right now."
            },
            "issue_summary": {
                "type": "string",
                "description": "A concise summary of the customer's issue and current status."
            },
            "intent_key": {
                "type": "string",
                "description": "The detected intent when known, such as technical-support."
            },
            "serial_number": {
                "type": "string",
                "description": "The product serial number if the user already provided it."
            }
        },
        "required": ["reason", "issue_summary"],
        "additionalProperties": False
    }
}

# Book CS search tool schema
BOOK_CS_CENTER_SCHEMA = {
    "type": "function",
    "name": "book_cs_center",
    "description": (
        "Book or change an appointment at a Contoso customer service center. "
        "Use this tool when the user wants to schedule a visit for device support or repair."
        "ALso use this tool when the user wants to change or cancel the visit appointment."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "customer_name": {
                "type": "string",
                "description": "The name of the customer booking the appointment."
            },
            "phone_number": {
                "type": "string",
                "description": "The contact phone number of the customer."
            },
            "device_model": {
                "type": "string",
                "description": "The Contoso device model for which service is requested (e.g., 'Contoso Phone Ultra', 'Contoso Phone Plus', 'Contoso Fold')."
            },
            "preferred_date": {
                "type": "string",
                "description": "The preferred appointment date in ISO 8601 format (e.g., '2024-04-18')."
            },
            "preferred_time": {
                "type": "string",
                "description": "The preferred appointment time (e.g., '14:00')."
            },
            "service_type": {
                "type": "string",
                "description": "The type of service needed (e.g., 'repair', 'consultation', 'maintenance')."
            },
            "preferred_location": {
                "type": "string",
                "enum": ["City hall store", "Ansan store", "Suwon store", "Seoul store"],
                "description": "The preferred customer service center location."
            }
        },
        "required": [
            "customer_name",
            "phone_number",
            "device_model",
            "preferred_date",
            "preferred_time",
            "service_type",
            "preferred_location"
        ],
        "additionalProperties": False
    }
}

# Bing search tool schema
IMAGE_CAPTION_SCHEMA = {
    "type": "function",
    "name": "image_caption",
    "description": (
        "Generate a descriptive caption for the current camera frame whenever the user asks about the scene."
        " Analyze the latest image from the camera and return a concise, relevant caption."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["single", "multiple"],
                "description": "Specify 'single' to analyze one frame or 'multiple' to analyze a batch of frames."
            },
            "target": {
                "type": "string",
                "enum": ["current_camera_frame"],
                "description": "Always set to 'current_camera_frame' to indicate the image source."
            },
            "timestamp": {
                "type": "string",
                "description": (
                    "The timestamp of the frame to analyze, in ISO 8601 format (e.g., '2024-04-16T12:34:56Z'). "
                    "Always use the latest frame timestamp available."
                )
            }
        },
        "required": ["method", "target", "timestamp"],
        "additionalProperties": False
    }
}

# Phone store search tool schema
PHONE_STORE_SEARCH_SCHEMA = {
    "type": "function",
    "name": "search_phone_store",
    "description": (
        "Search for phone stores. Only use this tool when users want to buy a device. "
        "Use this tool to find the store which sells specific phone models."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "phone_model": {
                "type": "string",
                "description": "The phone model to search for (e.g., 'galaxy s25 ultra', 'galaxy s25')"
            },
        },
        "required": ["phone_model"],
        "additionalProperties": False
    }
}

# Phone store search tool schema
ADD_TO_CART_SCHEMA = {
    "type": "function",
    "name": "add_to_cart",
    "description": "Add a phone to the shopping cart. Use this tool when a user wants to purchase or add a specific phone model to their cart.",
    "parameters": {
        "type": "object",
        "properties": {
            "phone_model": {
                "type": "string",
                "description": "The phone model to add to the cart (e.g., 'galaxy s25 ultra', 'galaxy s25 ')"
            },
        },
        "required": ["phone_model"],
        "additionalProperties": False
    }
}

# Internal knowledge base search tool schema
INTERNAL_SEARCH_SCHEMA = {
    "type": "function",
    "name": "internal_search",
    "description": "Search the knowledge base. The knowledge base is in Chinese, translate to and from Chinese if " +
                   "needed. Results are formatted as a source name first in square brackets, followed by the text " +
                   "content, and a line with '-----' at the end of each result.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

INTENT_SEARCH_SCHEMA = {
    "type": "function",
    "name": "intent_search",
    "description": (
        "Detect the user's support intent from the latest request and return an intent_key "
        "that the assistant can use to choose the correct workflow."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The latest user request or clarified issue to classify."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

# Grounding reporting tool schema
GROUNDING_REPORT_SCHEMA = {
    "type": "function",
    "name": "report_grounding",
    "description": "Report use of a source from the knowledge base as part of an answer (effectively, cite the source). Sources " +
                   "appear in square brackets before each knowledge base passage. Always use this tool to cite sources when responding " +
                   "with information from the knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of source names from last statement actually used, do not include the ones not used to formulate a response"
            }
        },
        "required": ["sources"],
        "additionalProperties": False
    }
}

REPORT_GROUNDING_SCHEMA = GROUNDING_REPORT_SCHEMA

# Send to Teams tool schema
SEND_TO_TEAMS_SCHEMA = {
    "type": "function",
    "name": "send_to_teams",
    "description": (
        "Send a message to Microsoft Teams via Logic App. "
        "Supports text-only messages, messages with images, or messages with charts (pie, bar, line, doughnut). "
        "Use this tool when the user wants to share information, notifications, reports, or visualizations to Teams."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The main text message to send to Teams. This is required."
            },
            "title": {
                "type": "string",
                "description": "Optional title for the message card. If not provided, a default title will be used."
            },
            "image_url": {
                "type": "string",
                "description": "Optional URL to an image or chart to include in the message. Can be a direct image URL or a chart URL generated by QuickChart."
            },
            "chart_config": {
                "type": "object",
                "description": "Optional chart configuration to generate a chart using QuickChart. If provided, a chart URL will be generated.",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["pie", "bar", "line", "doughnut"],
                        "description": "The type of chart to generate."
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels for the chart data points."
                    },
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Data values for the chart."
                    },
                    "chart_title": {
                        "type": "string",
                        "description": "Title for the chart (not the message title)."
                    },
                    "colors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional array of color codes for the chart. Defaults to predefined colors if not provided."
                    }
                },
                "required": ["chart_type", "labels", "data"]
            }
        },
        "required": ["message"],
        "additionalProperties": False
    }
}
