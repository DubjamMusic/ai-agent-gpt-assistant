#!/usr/bin/env python3
"""
GPT Agent - A simple OpenAI GPT-based agent implementation
"""

import os
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


class GPTAgent:
    """A simple GPT agent that can interact with users and perform tasks."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize the GPT Agent.
        
        Args:
            model: The GPT model to use
            api_key: OpenAI API key (if not provided, will look for OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.conversation_history: List[Dict[str, str]] = []
        
        if not self.api_key:
            print("Warning: No API key provided. Set OPENAI_API_KEY environment variable.")
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def chat(self, user_message: str) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            user_message: The user's message
            
        Returns:
            The agent's response
        """
        self.add_message("user", user_message)
        
        # TODO: Implement OpenAI API integration
        # This is a placeholder implementation. To make this functional:
        # 1. Import the openai library
        # 2. Initialize the OpenAI client with the API key
        # 3. Call the chat completion API with the conversation history
        # Example:
        #   from openai import OpenAI
        #   client = OpenAI(api_key=self.api_key)
        #   response = client.chat.completions.create(
        #       model=self.model,
        #       messages=self.conversation_history
        #   )
        #   return response.choices[0].message.content
        
        response = f"Agent received: {user_message}"
        self.add_message("assistant", response)
        
        return response
    
    def reset(self):
        """Reset the conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history


def main():
    """Main function to run the GPT agent in interactive mode."""
    print("GPT Agent - Interactive Mode")
    print("Type 'exit' or 'quit' to exit")
    print("-" * 50)
    
    agent = GPTAgent()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = agent.chat(user_input)
            print(f"Agent: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
