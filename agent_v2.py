#!/usr/bin/env python3
"""
GPT Agent v0.2 - Advanced Implementation with Modular Architecture

This is the refactored v0.2 implementation featuring:
- Modular tool system with plugin architecture
- Advanced memory management with summarization
- Async/await support for non-blocking operations
- Streaming response support
- Function calling (tool use) capabilities
"""

import os
import asyncio
from typing import List, Dict, Optional, Any, Callable, AsyncIterator
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI, AsyncOpenAI, APIError, APIConnectionError, RateLimitError
except ImportError:
    OpenAI = None
    AsyncOpenAI = None
    APIError = None
    APIConnectionError = None
    RateLimitError = None

from tools import ToolRegistry, create_default_registry
from memory import MemoryManager


class AgentMode(Enum):
    """Enum for different agent operation modes."""
    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"


class GPTAgentV2:
    """
    Advanced GPT Agent with modular architecture and extended capabilities.
    
    Features:
    - Tool/function calling support
    - Advanced memory management
    - Async/streaming support
    - Plugin-based extensibility
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_memory_persistence: bool = False,
        max_memory_turns: int = 50,
        mode: AgentMode = AgentMode.SYNC
    ):
        """
        Initialize the advanced GPT Agent.
        
        Args:
            model: The GPT model to use
            api_key: OpenAI API key
            system_prompt: System prompt for agent behavior
            tool_registry: Custom tool registry (uses default if None)
            enable_memory_persistence: Whether to persist conversation history
            max_memory_turns: Maximum conversation turns to keep in memory
            mode: Operation mode (sync, async, or streaming)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.mode = mode
        
        # Initialize clients
        self.client = None
        self.async_client = None
        self._initialize_clients()
        
        # Initialize tool registry
        self.tool_registry = tool_registry or create_default_registry()
        
        # Initialize memory manager
        self.memory = MemoryManager(
            max_turns=max_memory_turns,
            enable_persistence=enable_memory_persistence
        )
        
        if system_prompt:
            self.memory.set_system_message(system_prompt)
    
    def _initialize_clients(self) -> None:
        """Initialize OpenAI clients."""
        if not self.api_key:
            print(
                "Warning: No OpenAI API key provided. The agent will not be able to call "
                "the OpenAI API and any real GPT-based responses will not function.\n"
                "Set the OPENAI_API_KEY environment variable or pass api_key to GPTAgentV2.\n"
                "You can create an API key at: https://platform.openai.com/api-keys"
            )
        elif OpenAI is None:
            print("Warning: OpenAI library not installed. Install with: pip install openai")
        else:
            self.client = OpenAI(api_key=self.api_key)
            if AsyncOpenAI is not None:
                self.async_client = AsyncOpenAI(api_key=self.api_key)
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Optional[List] = None
    ) -> None:
        """
        Register a new tool for the agent to use.
        
        Args:
            name: Tool name
            func: Callable function
            description: Tool description
            parameters: Tool parameters
        """
        self.tool_registry.register(name, func, description, parameters)
    
    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response (synchronous).
        
        Args:
            user_message: The user's message
        
        Returns:
            The agent's response
        """
        self.memory.add_turn("user", user_message)
        
        if not self.client:
            response_text = f"Agent received: {user_message}"
            self.memory.add_turn("assistant", response_text)
            return response_text
        
        try:
            # Prepare messages with context window
            messages = self.memory.get_context_window()
            
            # Prepare tools for function calling
            tools = self.tool_registry.get_schemas() if self.tool_registry.list_tools() else None
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None
            )
            
            # Process response
            assistant_message = response.choices[0].message.content or ""
            
            # Handle tool calls if present
            if response.choices[0].message.tool_calls:
                assistant_message = self._handle_tool_calls(
                    response.choices[0].message.tool_calls,
                    assistant_message
                )
            
            self.memory.add_turn("assistant", assistant_message)
            return assistant_message
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            self.memory.add_turn("assistant", error_msg)
            return error_msg
    
    async def chat_async(self, user_message: str) -> str:
        """
        Send a message and get a response (asynchronous).
        
        Args:
            user_message: The user's message
        
        Returns:
            The agent's response
        """
        self.memory.add_turn("user", user_message)
        
        if not self.async_client:
            response_text = f"Agent received: {user_message}"
            self.memory.add_turn("assistant", response_text)
            return response_text
        
        try:
            messages = self.memory.get_context_window()
            tools = self.tool_registry.get_schemas() if self.tool_registry.list_tools() else None
            
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None
            )
            
            assistant_message = response.choices[0].message.content or ""
            
            if response.choices[0].message.tool_calls:
                assistant_message = self._handle_tool_calls(
                    response.choices[0].message.tool_calls,
                    assistant_message
                )
            
            self.memory.add_turn("assistant", assistant_message)
            return assistant_message
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            self.memory.add_turn("assistant", error_msg)
            return error_msg
    
    def stream_chat(self, user_message: str) -> AsyncIterator[str]:
        """
        Stream a response from the agent.
        
        Args:
            user_message: The user's message
        
        Yields:
            Chunks of the response as they arrive
        """
        self.memory.add_turn("user", user_message)
        
        if not self.client:
            yield f"Agent received: {user_message}"
            return
        
        try:
            messages = self.memory.get_context_window()
            
            with self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            ) as stream:
                full_response = ""
                for text in stream.text_stream:
                    full_response += text
                    yield text
                
                self.memory.add_turn("assistant", full_response)
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            self.memory.add_turn("assistant", error_msg)
            yield error_msg
    
    def _handle_tool_calls(self, tool_calls: List[Any], initial_response: str) -> str:
        """
        Handle tool calls from the agent.
        
        Args:
            tool_calls: List of tool calls from OpenAI
            initial_response: Initial response from the model
        
        Returns:
            Combined response with tool results
        """
        results = [initial_response] if initial_response else []
        
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.function.name
                tool_args = eval(tool_call.function.arguments)
                
                # Execute the tool
                result = self.tool_registry.call_tool(tool_name, **tool_args)
                results.append(f"\n[Tool: {tool_name}]\nResult: {result}")
                
            except Exception as e:
                results.append(f"\n[Tool Error]: {str(e)}")
        
        return "".join(results)
    
    def reset(self, keep_system_message: bool = True) -> None:
        """Reset the conversation memory."""
        self.memory.reset(keep_system_message=keep_system_message)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's memory usage."""
        return self.memory.get_statistics()
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return self.memory.get_history()
    
    def save_conversation(self, filepath: str) -> None:
        """Save the current conversation to a file."""
        self.memory.persistence_file = filepath
        self.memory.save_to_file()
    
    def load_conversation(self, filepath: str) -> None:
        """Load a conversation from a file."""
        self.memory.persistence_file = filepath
        self.memory.load_from_file()


# Backward compatibility wrapper
class GPTAgent:
    """
    Backward-compatible wrapper for v0.1 API.
    
    This maintains compatibility with existing code while using the v0.2 implementation.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """Initialize with v0.1 API."""
        self._agent = GPTAgentV2(
            model=model,
            api_key=api_key,
            system_prompt=system_prompt,
            mode=AgentMode.SYNC
        )
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to history."""
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Invalid role '{role}'")
        self._agent.memory.add_turn(role, content)
    
    def chat(self, user_message: str) -> str:
        """Chat with the agent."""
        return self._agent.chat(user_message)
    
    def reset(self) -> None:
        """Reset the conversation."""
        self._agent.reset()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        history = self._agent.memory.get_history()
        return [{"role": h["role"], "content": h["content"]} for h in history]


def main():
    """Main function to run the GPT agent in interactive mode."""
    print("GPT Agent v0.2 - Interactive Mode")
    print("Type 'exit' or 'quit' to exit")
    print("Type 'tools' to see available tools")
    print("Type 'stats' to see memory statistics")
    print("-" * 50)
    
    agent = GPTAgentV2()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "tools":
                tools = agent.tool_registry.list_tools()
                print(f"Available tools: {', '.join(tools)}")
                continue
            
            if user_input.lower() == "stats":
                stats = agent.get_memory_stats()
                print(f"Memory stats: {stats}")
                continue
            
            if not user_input:
                continue
            
            response = agent.chat(user_input)
            print(f"Agent: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
