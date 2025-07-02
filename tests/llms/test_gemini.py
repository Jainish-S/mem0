from unittest.mock import Mock, patch

import pytest

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.gemini import GeminiLLM


@pytest.fixture
def mock_gemini_client():
    with patch("mem0.llms.gemini.genai.Client") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        yield mock_client


def test_generate_response_without_tools(mock_gemini_client: Mock):
    config = BaseLlmConfig(model="gemini-2.0-flash-latest", temperature=0.7, max_tokens=100, top_p=1.0)
    llm = GeminiLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    # Create proper mock structure
    mock_part = Mock()
    mock_part.text = "I'm doing well, thank you for asking!"

    mock_content = Mock()
    mock_content.parts = [mock_part]

    mock_candidate = Mock()
    mock_candidate.content = mock_content

    mock_response = Mock()
    mock_response.candidates = [mock_candidate]

    mock_gemini_client.models.generate_content.return_value = mock_response

    response = llm.generate_response(messages)

    mock_gemini_client.models.generate_content.assert_called_once()

    assert response == "I'm doing well, thank you for asking!"


def test_generate_response_with_tools(mock_gemini_client: Mock):
    config = BaseLlmConfig(model="gemini-1.5-flash-latest", temperature=0.7, max_tokens=100, top_p=1.0)
    llm = GeminiLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Add a new memory: Today is a sunny day."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_memory",
                "description": "Add a memory",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "string", "description": "Data to add to memory"}},
                    "required": ["data"],
                },
            },
        }
    ]

    # Create proper mock structure for tool calls using a simple object
    class MockFunctionCall:
        def __init__(self, name, args):
            self.name = name
            self.args = args

    mock_tool_call = MockFunctionCall("add_memory", {"data": "Today is a sunny day."})

    mock_part_text = Mock()
    mock_part_text.text = "I've added the memory for you."
    # Explicitly ensure text part doesn't have function_call
    delattr(mock_part_text, "function_call") if hasattr(mock_part_text, "function_call") else None

    mock_part_function = Mock()
    mock_part_function.function_call = mock_tool_call
    # Remove text attribute from function part to avoid confusion
    delattr(mock_part_function, "text") if hasattr(mock_part_function, "text") else None

    mock_content = Mock()
    mock_content.parts = [mock_part_text, mock_part_function]

    mock_candidate = Mock()
    mock_candidate.content = mock_content

    mock_response = Mock()
    mock_response.candidates = [mock_candidate]

    mock_gemini_client.models.generate_content.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    mock_gemini_client.models.generate_content.assert_called_once()

    assert response["content"] == "I've added the memory for you."
    assert len(response["tool_calls"]) == 1
    assert response["tool_calls"][0]["name"] == "add_memory"
    assert response["tool_calls"][0]["arguments"] == {"data": "Today is a sunny day."}
