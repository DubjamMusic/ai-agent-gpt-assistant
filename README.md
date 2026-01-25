# GPT Agent

A simple, extensible GPT-based agent implementation for interacting with OpenAI's GPT models.

This project serves as a lightweight **agent kernel**—a foundational layer for building intelligent, interactive AI agents. It provides the essential infrastructure for conversation management, API integration, and extensible behavior configuration, making it easy to develop custom agent applications.

## Features

- 🤖 Interactive chat interface with real OpenAI API integration
- 💬 Conversation history management
- 🔧 Configurable GPT model selection
- 🔐 Secure API key management
- 🎯 Customizable system prompts for agent behavior
- 🐍 Pure Python implementation

## Roadmap (v0.2+)

The following enhancements are planned for future releases:

- **Multi-Agent Coordination**: Enable multiple agent instances to collaborate and share context
- **Tool Integration Framework**: Built-in support for function calling and external tool integration
- **Memory & Context Management**: Advanced conversation memory with summarization and retrieval
- **Streaming Responses**: Real-time streaming of agent responses for better UX
- **Agent Templates**: Pre-configured agent templates for common use cases (coding assistant, researcher, etc.)
- **Conversation Branching**: Support for exploring multiple conversation paths
- **Enhanced Error Handling**: Robust retry logic and fallback strategies
- **Performance Monitoring**: Built-in metrics and logging for agent performance analysis

## Limitations (Current)

The current implementation (v0.1) has the following limitations:

- **No Tool Calling**: The agent cannot execute functions or call external tools yet
- **Limited Context Window Management**: No automatic truncation or summarization of long conversations
- **Single Agent Only**: No support for multi-agent systems or agent coordination
- **Basic Error Handling**: Simple error messages without retry or fallback mechanisms
- **No Streaming**: Responses are received in full, not streamed in real-time
- **Memory Constraints**: All conversation history is kept in memory without persistence
- **No Async Support**: Synchronous API calls only, which may block on long-running requests

These limitations will be addressed in upcoming releases as outlined in the Roadmap.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DubjamMusic/ai-agent-gpt-assistant.git
cd ai-agent-gpt-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
cp .env.example .env
# Open .env in a text editor and replace 'your_api_key_here' with your actual OpenAI API key
# You can create and manage API keys at https://platform.openai.com/api-keys
```

## Usage

### Interactive Mode

Run the agent in interactive mode:
```bash
python agent.py
```

Then simply type your messages and press Enter to chat with the agent.

### Programmatic Usage

```python
from agent import GPTAgent

# Initialize the agent
agent = GPTAgent(model="gpt-3.5-turbo")

# Initialize with a custom system prompt
agent_with_prompt = GPTAgent(
    model="gpt-3.5-turbo",
    system_prompt="You are a helpful assistant that speaks like a pirate."
)

# Chat with the agent
response = agent.chat("Hello, how are you?")
print(response)

# Get conversation history
history = agent.get_history()

# Reset the conversation
agent.reset()
```

## Configuration

The agent can be configured through:
- Environment variables (`.env` file)
- Constructor parameters

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## Requirements

- Python 3.7+
- OpenAI Python library
- python-dotenv

**Note**: The dependencies listed in `requirements.txt` are optional. The agent can function in a limited capacity without the OpenAI library (for testing/development), and python-dotenv is only needed for `.env` file support.

## Agent v0.2 Design Plan (Concise)

The v0.2 release will focus on modularity and extensibility:

### Core Architecture
- **Plugin System**: Modular architecture for adding capabilities (tools, memory backends, etc.)
- **Agent Base Class**: Abstract base for creating specialized agent types
- **Event System**: Publish-subscribe pattern for agent lifecycle events

### Key Components
1. **Tool Registry**: Centralized registry for function calling and external tool integration
2. **Memory Manager**: Pluggable memory backends (in-memory, file-based, vector stores)
3. **Prompt Template Engine**: Reusable prompt templates with variable substitution
4. **Response Parser**: Structured output parsing for function calls and JSON responses

### API Changes
- Introduce async variants of core methods (`async_chat`, `async_stream`)
- Add configuration object for cleaner agent initialization
- Support for conversation serialization/deserialization

### Backward Compatibility
- v0.1 API will remain supported via compatibility layer
- Migration guide will be provided for existing users

## Project Structure

```
ai-agent-gpt-assistant/
├── agent.py           # Main agent implementation
├── requirements.txt   # Python dependencies
├── .env.example       # Example environment configuration
├── .gitignore        # Git ignore rules
└── README.md         # This file
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- [HustleCodex](https://github.com/DubjamMusic/hustlecodex) - HustleCodex V3 Reality Recovery Playing Game