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

try:
    from openai import OpenAI, APIError, APIConnectionError, RateLimitError
except ImportError:
    OpenAI = None  # OpenAI library not installed
    APIError = None
    APIConnectionError = None
    RateLimitError = None


class GPTAgent:
    """A simple GPT agent that can interact with users and perform tasks."""
    
    # Valid OpenAI chat message roles
    VALID_ROLES = {"user", "assistant", "system"}
    
    def __init__(
        self, 
        model: str = "gpt-3.5-turbo", 
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the GPT Agent.
        
        Args:
            model: The GPT model to use
            api_key: OpenAI API key (if not provided, will look for OPENAI_API_KEY env var)
            system_prompt: Optional system prompt to configure agent behavior
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.conversation_history: List[Dict[str, str]] = []
        self.client = None
        
        # Initialize conversation history with system prompt if provided
        if system_prompt:
            self.conversation_history.append({"role": "system", "content": system_prompt})
        
        if not self.api_key:
            print(
                "Warning: No OpenAI API key provided. The agent will not be able to call "
                "the OpenAI API and any real GPT-based responses will not function.\n"
                "Set the OPENAI_API_KEY environment variable or pass api_key to GPTAgent.\n"
                "You can create an API key at: https://platform.openai.com/api-keys"
            )
        elif OpenAI is None:
            print("Warning: OpenAI library not installed. Install with: pip install openai")
        else:
            self.client = OpenAI(api_key=self.api_key)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        # Validate role parameter
        if role not in self.VALID_ROLES:
            raise ValueError(
                f"Invalid role '{role}'. Must be one of: {', '.join(self.VALID_ROLES)}"
            )
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
        
        # If OpenAI client is available, use it to get a real response
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history
                )
                # Validate response has choices and content
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty response from OpenAI API")
                
                assistant_message = response.choices[0].message.content
                self.add_message("assistant", assistant_message)
                return assistant_message
            except (APIError, APIConnectionError, RateLimitError) as e:
                # Handle specific OpenAI API errors
                error_msg = f"OpenAI API error: {e}"
                print(error_msg)
                response_text = f"Agent received: {user_message} (API error)"
                self.add_message("assistant", response_text)
                return response_text
            except Exception as e:
                # Handle any other unexpected errors
                error_msg = f"Unexpected error: {e}"
                print(error_msg)
                response_text = f"Agent received: {user_message} (error occurred)"
                self.add_message("assistant", response_text)
                return response_text
        else:
            # Fallback for when client is not initialized (no API key or library not installed)
            response_text = f"Agent received: {user_message}"
            self.add_message("assistant", response_text)
            return response_text
    
    def reset(self):
        """Reset the conversation history, preserving system prompts."""
        # Keep system messages if they exist
        system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
        self.conversation_history = system_messages
    
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
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
