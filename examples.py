#!/usr/bin/env python3
"""
Examples demonstrating v0.2 features of the GPT Agent.

This module shows how to use:
- Tool registration and calling
- Memory management
- Async operations
- Streaming responses
"""

import asyncio
from typing import List
from agent_v2 import GPTAgentV2, GPTAgent, AgentMode
from tools import ToolRegistry, ToolParameter, create_default_registry
from memory import MemoryManager


# Example 1: Basic usage with tools
def example_basic_with_tools():
    """Example: Using the agent with built-in and custom tools."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage with Tools")
    print("="*60)
    
    # Create agent with default tools
    agent = GPTAgentV2(system_prompt="You are a helpful assistant.")
    
    # Register a custom tool
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}! Nice to meet you."
    
    agent.register_tool(
        name="greet",
        func=greet,
        description="Greet someone by their name",
        parameters=[
            ToolParameter(
                name="name",
                type="string",
                description="The name of the person to greet",
                required=True
            )
        ]
    )
    
    # List available tools
    print(f"\nAvailable tools: {agent.tool_registry.list_tools()}")
    
    # Chat with the agent
    response = agent.chat("What tools do you have available?")
    print(f"\nAgent response: {response}")


# Example 2: Memory management with persistence
def example_memory_management():
    """Example: Using advanced memory management."""
    print("\n" + "="*60)
    print("Example 2: Memory Management")
    print("="*60)
    
    # Create agent with memory persistence
    agent = GPTAgentV2(
        system_prompt="You are a knowledgeable assistant.",
        enable_memory_persistence=True,
        max_memory_turns=10
    )
    
    # Simulate a conversation
    messages = [
        "What is Python?",
        "Tell me about machine learning",
        "How does neural networks work?"
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        response = agent.chat(msg)
        print(f"Agent: {response[:100]}...")  # Show first 100 chars
    
    # Get memory statistics
    stats = agent.get_memory_stats()
    print(f"\nMemory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save conversation
    agent.save_conversation("example_conversation.json")
    print("\nConversation saved to 'example_conversation.json'")


# Example 3: Custom tool registry
def example_custom_tools():
    """Example: Creating and using a custom tool registry."""
    print("\n" + "="*60)
    print("Example 3: Custom Tool Registry")
    print("="*60)
    
    # Create custom registry
    registry = ToolRegistry()
    
    # Define custom tools
    def temperature_converter(celsius: float) -> str:
        """Convert Celsius to Fahrenheit."""
        fahrenheit = (celsius * 9/5) + 32
        return f"{celsius}°C = {fahrenheit}°F"
    
    def word_counter(text: str) -> str:
        """Count words in text."""
        count = len(text.split())
        return f"Word count: {count}"
    
    # Register tools
    registry.register(
        name="convert_temp",
        func=temperature_converter,
        description="Convert temperature from Celsius to Fahrenheit",
        parameters=[
            ToolParameter(
                name="celsius",
                type="number",
                description="Temperature in Celsius",
                required=True
            )
        ]
    )
    
    registry.register(
        name="count_words",
        func=word_counter,
        description="Count the number of words in text",
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="Text to count words in",
                required=True
            )
        ]
    )
    
    # Create agent with custom registry
    agent = GPTAgentV2(tool_registry=registry)
    
    print(f"\nRegistered tools: {registry.list_tools()}")
    
    # Test tool calling
    print("\nTesting tools:")
    print(f"  convert_temp(25): {registry.call_tool('convert_temp', celsius=25)}")
    print(f"  count_words('hello world test'): {registry.call_tool('count_words', text='hello world test')}")


# Example 4: Async operations
async def example_async_chat():
    """Example: Using async chat operations."""
    print("\n" + "="*60)
    print("Example 4: Async Chat Operations")
    print("="*60)
    
    agent = GPTAgentV2(
        system_prompt="You are a concise assistant.",
        mode=AgentMode.ASYNC
    )
    
    # Simulate multiple concurrent chat operations
    messages = [
        "What is AI?",
        "Explain blockchain",
        "What is quantum computing?"
    ]
    
    print("\nRunning async chat operations...")
    
    # Create tasks for concurrent execution
    tasks = [agent.chat_async(msg) for msg in messages]
    
    # Run all tasks concurrently
    responses = await asyncio.gather(*tasks)
    
    for msg, response in zip(messages, responses):
        print(f"\nUser: {msg}")
        print(f"Agent: {response[:100]}...")


# Example 5: Backward compatibility with v0.1 API
def example_backward_compatibility():
    """Example: Using v0.1 API for backward compatibility."""
    print("\n" + "="*60)
    print("Example 5: Backward Compatibility (v0.1 API)")
    print("="*60)
    
    # Create agent using v0.1 API
    agent = GPTAgent(system_prompt="You are a helpful assistant.")
    
    # Use v0.1 methods
    agent.add_message("user", "Hello!")
    response = agent.chat("How are you?")
    
    print(f"\nResponse: {response}")
    
    # Get history using v0.1 API
    history = agent.get_history()
    print(f"\nConversation history ({len(history)} messages):")
    for msg in history:
        print(f"  {msg['role']}: {msg['content'][:50]}...")


# Example 6: Memory summarization
def example_memory_summarization():
    """Example: Demonstrating automatic memory summarization."""
    print("\n" + "="*60)
    print("Example 6: Memory Summarization")
    print("="*60)
    
    # Create memory manager with low threshold for demo
    memory = MemoryManager(
        max_turns=10,
        summary_threshold=5,
        enable_persistence=False
    )
    
    memory.set_system_message("You are a helpful assistant.")
    
    # Add many turns to trigger summarization
    print("\nAdding conversation turns...")
    for i in range(15):
        memory.add_turn("user", f"Question {i+1}: Tell me about topic {i+1}")
        memory.add_turn("assistant", f"Here's information about topic {i+1}...")
    
    # Check memory state
    print(f"\nActive turns: {len(memory.turns)}")
    print(f"Summaries created: {len(memory.summaries)}")
    
    if memory.summaries:
        print(f"\nFirst summary: {memory.summaries[0][:100]}...")
    
    # Get context window
    context = memory.get_context_window()
    print(f"\nContext window messages: {len(context)}")


# Example 7: Tool schemas for API integration
def example_tool_schemas():
    """Example: Viewing tool schemas for API integration."""
    print("\n" + "="*60)
    print("Example 7: Tool Schemas")
    print("="*60)
    
    registry = create_default_registry()
    
    print("\nTool schemas for OpenAI API:")
    schemas = registry.get_schemas()
    
    import json
    for schema in schemas:
        print(f"\n{json.dumps(schema, indent=2)}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("GPT Agent v0.2 - Feature Examples")
    print("="*60)
    
    # Note: Some examples require API key
    print("\nNote: Some examples require OPENAI_API_KEY environment variable")
    
    # Run examples that don't require API
    try:
        example_basic_with_tools()
    except Exception as e:
        print(f"Skipped (requires API): {e}")
    
    try:
        example_memory_management()
    except Exception as e:
        print(f"Skipped (requires API): {e}")
    
    example_custom_tools()
    example_backward_compatibility()
    example_memory_summarization()
    example_tool_schemas()
    
    # Run async example
    print("\n" + "="*60)
    print("Running async example...")
    print("="*60)
    try:
        asyncio.run(example_async_chat())
    except Exception as e:
        print(f"Skipped (requires API): {e}")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
