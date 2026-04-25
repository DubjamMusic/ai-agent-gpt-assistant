#!/usr/bin/env python3
"""
Tool Registry System for GPT Agent

This module provides a modular tool system that allows registering and managing
functions that the GPT agent can call. Tools are organized in a registry and can
be dynamically added, removed, and invoked by the agent.
"""

import inspect
import json
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ToolParameter:
    """Represents a parameter for a tool function."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolSchema:
    """Schema definition for a tool that can be called by the agent."""
    name: str
    description: str
    parameters: List[ToolParameter]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool schema to dictionary format for OpenAI API."""
        properties = {}
        required_params = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.required:
                required_params.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params
                }
            }
        }


class ToolRegistry:
    """
    Registry for managing tools that the GPT agent can call.
    
    This allows for dynamic tool registration and provides a centralized
    way to manage all available functions the agent can execute.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, ToolSchema] = {}
    
    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Optional[List[ToolParameter]] = None
    ) -> None:
        """
        Register a tool (function) in the registry.
        
        Args:
            name: The name of the tool
            func: The callable function to register
            description: A description of what the tool does
            parameters: List of ToolParameter objects describing the function's parameters
        
        Raises:
            ValueError: If a tool with the same name already exists
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        
        self._tools[name] = func
        
        # Auto-generate parameters from function signature if not provided
        if parameters is None:
            parameters = self._extract_parameters(func)
        
        schema = ToolSchema(name=name, description=description, parameters=parameters)
        self._schemas[name] = schema
    
    def unregister(self, name: str) -> None:
        """
        Unregister a tool from the registry.
        
        Args:
            name: The name of the tool to unregister
        
        Raises:
            KeyError: If the tool is not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        
        del self._tools[name]
        del self._schemas[name]
    
    def get_tool(self, name: str) -> Callable:
        """
        Get a tool function by name.
        
        Args:
            name: The name of the tool
        
        Returns:
            The callable function
        
        Raises:
            KeyError: If the tool is not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]
    
    def call_tool(self, name: str, **kwargs) -> Any:
        """
        Call a registered tool with the given arguments.
        
        Args:
            name: The name of the tool to call
            **kwargs: Arguments to pass to the tool
        
        Returns:
            The result of the tool execution
        
        Raises:
            KeyError: If the tool is not found
            TypeError: If the arguments don't match the tool's signature
        """
        tool = self.get_tool(name)
        return tool(**kwargs)
    
    def list_tools(self) -> List[str]:
        """Get a list of all registered tool names."""
        return list(self._tools.keys())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all tool schemas in OpenAI function calling format.
        
        Returns:
            List of tool schemas formatted for OpenAI API
        """
        return [schema.to_dict() for schema in self._schemas.values()]
    
    def get_schema(self, name: str) -> Dict[str, Any]:
        """
        Get a specific tool schema by name.
        
        Args:
            name: The name of the tool
        
        Returns:
            The tool schema in OpenAI format
        
        Raises:
            KeyError: If the tool is not found
        """
        if name not in self._schemas:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._schemas[name].to_dict()
    
    @staticmethod
    def _extract_parameters(func: Callable) -> List[ToolParameter]:
        """
        Extract parameters from a function signature.
        
        Args:
            func: The function to extract parameters from
        
        Returns:
            List of ToolParameter objects
        """
        sig = inspect.signature(func)
        parameters = []
        
        for param_name, param in sig.parameters.items():
            # Skip 'self' parameter for methods
            if param_name == 'self':
                continue
            
            # Determine parameter type from annotation
            param_type = "string"  # Default type
            if param.annotation != inspect.Parameter.empty:
                annotation = param.annotation
                if annotation == int:
                    param_type = "integer"
                elif annotation == float:
                    param_type = "number"
                elif annotation == bool:
                    param_type = "boolean"
                elif annotation == list:
                    param_type = "array"
                elif annotation == dict:
                    param_type = "object"
            
            # Determine if parameter is required
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            
            # Use docstring or parameter name as description
            description = f"Parameter: {param_name}"
            
            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=description,
                required=required,
                default=default
            ))
        
        return parameters


# Built-in tools for the agent
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression to evaluate
    
    Returns:
        The result of the calculation
    """
    try:
        # Only allow safe mathematical operations
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow, 'sqrt': lambda x: x ** 0.5
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def web_search(query: str) -> str:
    """
    Simulate a web search (placeholder for future integration).
    
    Args:
        query: The search query
    
    Returns:
        Search results
    """
    return f"Web search for '{query}' would be performed here (not yet implemented)"


def create_default_registry() -> ToolRegistry:
    """
    Create a registry with default built-in tools.
    
    Returns:
        A ToolRegistry with common tools pre-registered
    """
    registry = ToolRegistry()
    
    # Register built-in tools
    registry.register(
        name="get_time",
        func=get_current_time,
        description="Get the current date and time"
    )
    
    registry.register(
        name="calculate",
        func=calculate,
        description="Evaluate a mathematical expression",
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                description="A mathematical expression to evaluate",
                required=True
            )
        ]
    )
    
    registry.register(
        name="web_search",
        func=web_search,
        description="Search the web for information",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="The search query",
                required=True
            )
        ]
    )
    
    return registry
