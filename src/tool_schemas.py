from __future__ import annotations

DEFAULT_MODE = "user"
DEV_MODE = "dev"
USER_MODE = "user"

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

DEV_TOOLS = [BASH_TOOL, PYTHON_TOOL, WEB_SEARCH_TOOL, FETCH_URL_TOOL, GET_LOGS_TOOL]
USER_TOOLS = [PYTHON_TOOL, WEB_SEARCH_TOOL, FETCH_URL_TOOL]


def get_tools_for_mode(mode: str) -> list[dict]:
    if mode == DEV_MODE:
        return DEV_TOOLS
    return USER_TOOLS
