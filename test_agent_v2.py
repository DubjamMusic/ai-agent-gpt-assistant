#!/usr/bin/env python3
"""
Test suite for GPT Agent v0.2

Tests cover:
- Tool registry functionality
- Memory management
- Agent chat operations
- Backward compatibility
"""

import unittest
import json
import os
import tempfile
from tools import ToolRegistry, ToolParameter, create_default_registry
from memory import MemoryManager, ConversationTurn
from agent_v2 import GPTAgentV2, GPTAgent, AgentMode


class TestToolRegistry(unittest.TestCase):
    """Tests for the tool registry system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()
    
    def test_register_tool(self):
        """Test registering a tool."""
        def dummy_tool(x: int) -> int:
            return x * 2
        
        self.registry.register(
            name="double",
            func=dummy_tool,
            description="Double a number"
        )
        
        self.assertIn("double", self.registry.list_tools())
    
    def test_register_duplicate_tool(self):
        """Test that registering duplicate tool raises error."""
        def tool1():
            pass
        
        self.registry.register("tool", tool1, "Tool 1")
        
        with self.assertRaises(ValueError):
            self.registry.register("tool", tool1, "Tool 1")
    
    def test_call_tool(self):
        """Test calling a registered tool."""
        def add(a: int, b: int) -> int:
            return a + b
        
        self.registry.register("add", add, "Add two numbers")
        result = self.registry.call_tool("add", a=2, b=3)
        
        self.assertEqual(result, 5)
    
    def test_call_nonexistent_tool(self):
        """Test calling a non-existent tool raises error."""
        with self.assertRaises(KeyError):
            self.registry.call_tool("nonexistent")
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        def tool():
            pass
        
        self.registry.register("test", tool, "Test tool")
        self.assertIn("test", self.registry.list_tools())
        
        self.registry.unregister("test")
        self.assertNotIn("test", self.registry.list_tools())
    
    def test_get_schemas(self):
        """Test getting tool schemas for API."""
        def multiply(a: int, b: int) -> int:
            return a * b
        
        self.registry.register(
            name="multiply",
            func=multiply,
            description="Multiply two numbers",
            parameters=[
                ToolParameter("a", "integer", "First number"),
                ToolParameter("b", "integer", "Second number")
            ]
        )
        
        schemas = self.registry.get_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "multiply")
    
    def test_default_registry(self):
        """Test creating default registry with built-in tools."""
        registry = create_default_registry()
        tools = registry.list_tools()
        
        self.assertIn("get_time", tools)
        self.assertIn("calculate", tools)
        self.assertIn("web_search", tools)


class TestMemoryManager(unittest.TestCase):
    """Tests for the memory management system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory = MemoryManager(max_turns=10, summary_threshold=5)
    
    def test_add_turn(self):
        """Test adding a conversation turn."""
        self.memory.add_turn("user", "Hello")
        self.assertEqual(len(self.memory.turns), 1)
        self.assertEqual(self.memory.turns[0].role, "user")
    
    def test_set_system_message(self):
        """Test setting system message."""
        self.memory.set_system_message("You are helpful")
        self.assertEqual(self.memory.system_message, "You are helpful")
    
    def test_get_context_window(self):
        """Test getting context window."""
        self.memory.set_system_message("System prompt")
        self.memory.add_turn("user", "Hello")
        self.memory.add_turn("assistant", "Hi there")
        
        context = self.memory.get_context_window()
        
        # Should have system message + 2 turns
        self.assertGreaterEqual(len(context), 2)
        self.assertEqual(context[0]["role"], "system")
    
    def test_reset(self):
        """Test resetting memory."""
        self.memory.set_system_message("System")
        self.memory.add_turn("user", "Hello")
        
        self.memory.reset(keep_system_message=True)
        
        self.assertEqual(len(self.memory.turns), 0)
        self.assertEqual(self.memory.system_message, "System")
    
    def test_reset_without_system(self):
        """Test resetting memory without keeping system message."""
        self.memory.set_system_message("System")
        self.memory.add_turn("user", "Hello")
        
        self.memory.reset(keep_system_message=False)
        
        self.assertEqual(len(self.memory.turns), 0)
        self.assertIsNone(self.memory.system_message)
    
    def test_persistence(self):
        """Test saving and loading conversation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
        
        try:
            # Create memory and add data
            memory1 = MemoryManager(persistence_file=temp_file)
            memory1.set_system_message("Test system")
            memory1.add_turn("user", "Hello")
            memory1.add_turn("assistant", "Hi")
            memory1.save_to_file()
            
            # Load in new memory instance
            memory2 = MemoryManager(persistence_file=temp_file)
            memory2.load_from_file()
            
            self.assertEqual(memory2.system_message, "Test system")
            self.assertEqual(len(memory2.turns), 2)
            self.assertEqual(memory2.turns[0].content, "Hello")
        
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_statistics(self):
        """Test getting memory statistics."""
        self.memory.add_turn("user", "Question 1")
        self.memory.add_turn("assistant", "Answer 1")
        self.memory.add_turn("user", "Question 2")
        
        stats = self.memory.get_statistics()
        
        self.assertEqual(stats["total_turns"], 3)
        self.assertEqual(stats["user_turns"], 2)
        self.assertEqual(stats["assistant_turns"], 1)
        self.assertGreater(stats["total_characters"], 0)


class TestGPTAgentV2(unittest.TestCase):
    """Tests for GPTAgentV2."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create agent without API key for testing
        self.agent = GPTAgentV2(api_key=None)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.model, "gpt-3.5-turbo")
        self.assertIsNotNone(self.agent.tool_registry)
        self.assertIsNotNone(self.agent.memory)
    
    def test_register_tool(self):
        """Test registering a tool with agent."""
        def test_tool(x: int) -> int:
            return x * 2
        
        self.agent.register_tool("test", test_tool, "Test tool")
        self.assertIn("test", self.agent.tool_registry.list_tools())
    
    def test_memory_stats(self):
        """Test getting memory statistics."""
        self.agent.memory.add_turn("user", "Hello")
        self.agent.memory.add_turn("assistant", "Hi")
        
        stats = self.agent.get_memory_stats()
        
        self.assertEqual(stats["total_turns"], 2)
        self.assertEqual(stats["user_turns"], 1)
        self.assertEqual(stats["assistant_turns"], 1)
    
    def test_reset(self):
        """Test resetting agent."""
        self.agent.memory.add_turn("user", "Hello")
        self.agent.reset()
        
        self.assertEqual(len(self.agent.memory.turns), 0)
    
    def test_save_load_conversation(self):
        """Test saving and loading conversation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
        
        try:
            # Add some conversation
            self.agent.memory.add_turn("user", "Test message")
            self.agent.save_conversation(temp_file)
            
            # Create new agent and load
            agent2 = GPTAgentV2(api_key=None)
            agent2.load_conversation(temp_file)
            
            self.assertEqual(len(agent2.memory.turns), 1)
            self.assertEqual(agent2.memory.turns[0].content, "Test message")
        
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestBackwardCompatibility(unittest.TestCase):
    """Tests for v0.1 API backward compatibility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = GPTAgent(api_key=None)
    
    def test_add_message(self):
        """Test adding message with v0.1 API."""
        self.agent.add_message("user", "Hello")
        history = self.agent.get_history()
        
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "user")
    
    def test_invalid_role(self):
        """Test that invalid role raises error."""
        with self.assertRaises(ValueError):
            self.agent.add_message("invalid", "Hello")
    
    def test_get_history(self):
        """Test getting history with v0.1 API."""
        self.agent.add_message("user", "Q1")
        self.agent.add_message("assistant", "A1")
        
        history = self.agent.get_history()
        
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["content"], "Q1")
        self.assertEqual(history[1]["content"], "A1")
    
    def test_reset(self):
        """Test reset with v0.1 API."""
        self.agent.add_message("user", "Hello")
        self.agent.reset()
        
        history = self.agent.get_history()
        self.assertEqual(len(history), 0)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
