import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from mcp import ClientSession
from openai import AsyncOpenAI
from anthropic import Anthropic


async def build_messages_and_tools(
    query: str, session: ClientSession
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Common helper to turn a user query + MCP session into
    (messages, tools) for any provider.
    """
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": query,
        }
    ]

    tools_response = await session.list_tools()
    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
        for tool in tools_response.tools
    ]
    return messages, tools


class BaseLLMClient(ABC):
    """
    Abstract base for LLM providers.
    Subclasses implement provider-specific tool-calling behaviour.
    """

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    async def process_query(self, query: str, session: ClientSession) -> str:
        """
        Run a query through this LLM with MCP tools available via the session.
        """
        raise NotImplementedError


class OpenAILLMClient(BaseLLMClient):
    def __init__(self, model: str | None = None) -> None:
        super().__init__(model or "gpt-4.1")
        self.client = AsyncOpenAI()

    async def process_query(self, query: str, session: ClientSession) -> str:
        messages, available_tools = await build_messages_and_tools(query, session)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": tool,
                }
                for tool in available_tools
            ],
            tool_choice="auto",
            max_tokens=1000,
        )

        final_text: List[str] = []

        while True:
            choice = response.choices[0]
            message = choice.message

            # Append assistant text, if any
            if message.content:
                final_text.append(
                    "".join(
                        part["text"] if isinstance(part, dict) and part.get("type") == "text" else str(part)
                        for part in (message.content if isinstance(message.content, list) else [message.content])
                    )
                )

            # No tool calls → we're done
            if not message.tool_calls:
                break

            # Handle tool calls from OpenAI
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                raw_args = tool_call.function.arguments
                # OpenAI returns JSON-encoded arguments; MCP expects a dict
                if isinstance(raw_args, str):
                    try:
                        tool_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        # Fall back to empty dict on bad JSON, to avoid crashing
                        tool_args = {}
                else:
                    tool_args = raw_args

                # Execute tool call on MCP server
                result = await session.call_tool(tool_name, tool_args)
                final_text.append(f"Calling {tool_name} with input: {tool_args}")

                # Add the tool result back into the conversation
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [t.model_dump() for t in message.tool_calls],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": str(result.content),
                    }
                )

            # Get next response from OpenAI with updated messages
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[
                    {
                        "type": "function",
                        "function": tool,
                    }
                    for tool in available_tools
                ],
                tool_choice="auto",
                max_tokens=1000,
            )

        return "\n".join(final_text)


class AnthropicLLMClient(BaseLLMClient):
    def __init__(self, model: str | None = None) -> None:
        super().__init__(model or "claude-3-5-sonnet-latest")
        self.client = Anthropic()

    async def process_query(self, query: str, session: ClientSession) -> str:
        messages, available_tools = await build_messages_and_tools(query, session)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
        )

        final_text: List[str] = []

        for content in response.content:
            if content.type == "text":
                final_text.append(content.text)
            elif content.type == "tool_use":
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await session.call_tool(tool_name, tool_args)
                final_text.append(f"Calling {tool_name} with input: {tool_args}")

        return "\n".join(final_text)


def create_llm_client() -> BaseLLMClient:
    """
    Factory that returns the right concrete LLM client based on env config.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL")

    if provider == "openai":
        return OpenAILLMClient(model)
    if provider == "anthropic":
        return AnthropicLLMClient(model)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
