# GPT Agent

A simple, extensible GPT-based agent implementation for interacting with OpenAI's GPT models.

## Features

- 🤖 Interactive chat interface with real OpenAI API integration
- 💬 Conversation history management
- 🔧 Configurable GPT model selection
- 🔐 Secure API key management
- 🎯 Customizable system prompts for agent behavior
- 🐍 Pure Python implementation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DubjamMusic/Gpt-agent.git
cd Gpt-agent
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

## Project Structure

```
Gpt-agent/
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