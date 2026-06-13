from __future__ import annotations

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command and return stdout/stderr and exit code.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (1-120).",
                    "minimum": 1,
                    "maximum": 120,
                    "default": 20,
                },
            },
            "required": ["command"],
        },
    },
}

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run a Python code snippet and return stdout/stderr and exit code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (1-120).",
                    "minimum": 1,
                    "maximum": 120,
                    "default": 20,
                },
            },
            "required": ["code"],
        },
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information and return concise results with URLs.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-10).",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

FETCH_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch a URL and return readable page text for deeper analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The absolute URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum number of text characters to return (500-20000).",
                    "minimum": 500,
                    "maximum": 20000,
                    "default": 8000,
                },
            },
            "required": ["url"],
        },
    },
}

GET_LOGS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_logs",
        "description": "Get recent logs from the frontend or backend runtime.",
        "parameters": {
            "type": "object",
            "properties": {
                "system": {
                    "type": "string",
                    "description": "Which runtime to get logs from.",
                    "enum": ["frontend", "backend"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of recent log lines to return (1-500).",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 50,
                },
            },
            "required": ["system"],
        },
    },
}

PUBLISH_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "publish_document",
        "description": (
            "Convert a Markdown document into a PDF, store it, and return a secure "
            "download link. Use this to deliver long-form documents — such as a "
            "research report — to the user. Pass the complete document as Markdown."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "markdown": {
                    "type": "string",
                    "description": "The full document content in Markdown.",
                },
                "title": {
                    "type": "string",
                    "description": "Optional title shown in the PDF.",
                },
            },
            "required": ["markdown"],
        },
    },
}

# --- home control: operate the house through the hub API --------------------

HOME_CAPABILITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "home_capabilities",
        "description": (
            "Discover what the smart home can do: device kinds, every device "
            "(id, name, kind) and its available actions, the scenes, and the "
            "audio stations. Call this first when you need to control something "
            "and don't already know the device id or action."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

HOME_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "home_status",
        "description": (
            "Get the current home overview: who is home (presence), internet/"
            "network health, and any anomalies."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

LIST_DEVICES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_devices",
        "description": "List all smart-home devices with their live status (e.g. blind position/state).",
        "parameters": {"type": "object", "properties": {}},
    },
}

CONTROL_DEVICE_TOOL = {
    "type": "function",
    "function": {
        "name": "control_device",
        "description": (
            "Run an action on a smart-home device by its id (e.g. open/close/stop "
            "a blind, on/off a relay, or set a position). Use home_capabilities or "
            "list_devices to find device ids and valid actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string", "description": "Device id, e.g. 'wz-gross'."},
                "action": {"type": "string", "description": "Action, e.g. 'open', 'close', 'stop', 'position'."},
                "params": {
                    "type": "object",
                    "description": "Optional action params, e.g. {\"value\": 30} for a position 0..100.",
                },
            },
            "required": ["device_id", "action"],
        },
    },
}

RUN_SCENE_TOOL = {
    "type": "function",
    "function": {
        "name": "run_scene",
        "description": "Run a named home scene (e.g. morning, evening, night, leaving, deter).",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Scene name."},
            },
            "required": ["name"],
        },
    },
}

CONTROL_AUDIO_TOOL = {
    "type": "function",
    "function": {
        "name": "control_audio",
        "description": "Control room audio: play a radio station, stop, or set volume (0..1).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["play", "stop", "volume"]},
                "station": {"type": "string", "description": "Station name for action=play (e.g. 'Flux FM')."},
                "volume": {"type": "number", "description": "Volume 0..1 for action=volume.", "minimum": 0, "maximum": 1},
            },
            "required": ["action"],
        },
    },
}

HOME_TOOLS = [
    HOME_CAPABILITIES_TOOL,
    HOME_STATUS_TOOL,
    LIST_DEVICES_TOOL,
    CONTROL_DEVICE_TOOL,
    RUN_SCENE_TOOL,
    CONTROL_AUDIO_TOOL,
]

ALL_TOOLS = [
    WEB_SEARCH_TOOL,
    FETCH_URL_TOOL,
    PYTHON_TOOL,
    BASH_TOOL,
    GET_LOGS_TOOL,
    PUBLISH_DOCUMENT_TOOL,
    *HOME_TOOLS,
]


def get_tools() -> list[dict]:
    """Return the full tool set available to the assistant."""
    return ALL_TOOLS
