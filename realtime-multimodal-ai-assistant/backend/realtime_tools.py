import json
from enum import Enum
from typing import Any, Callable


class ToolResultDirection(Enum):
    TO_SERVER = 1
    TO_CLIENT = 2


class ToolResult:
    text: str
    destination: ToolResultDirection

    def __init__(self, text: str, destination: ToolResultDirection):
        self.text = text
        self.destination = destination

    def to_text(self) -> str:
        if self.text is None:
            return ""
        return self.text if isinstance(self.text, str) else json.dumps(self.text)


class Tool:
    target: Callable[..., ToolResult]
    schema: Any

    def __init__(self, target: Any, schema: Any):
        self.target = target
        self.schema = schema