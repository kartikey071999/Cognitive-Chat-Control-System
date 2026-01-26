# Cognitive Chat Control System

A sophisticated AI assistant (conceptually powered by GPT-2 for demonstration) that features emotional state tracking, intent recognition, and long-term memory management.

## 🧠 Features

-   **Modular Architecture**: Clean separation of concerns (Logic, State, Configuration, LLM).
-   **Intent Recognition**: Classifies user input into intents like `question`, `instruction`, `emotional_support`, etc.
-   **Emotional Engine**: Tracks and updates emotional states (`neutral`, `curious`, `confident`, `empathetic`, `frustrated`) based on interactions.
-   **Memory Management**: Maintains a working memory and summarizes long-term facts when memory limits are reached.
-   **Configurable**: Easy to adjust model parameters and generation settings.

## 📂 Project Structure

```
Cognitive-Chat-Control-System/
├── main.py            # Entry point
├── src/
│   ├── config.py      # Configuration constants
│   ├── llm.py         # LLM initialization and wrapper
│   ├── state.py       # State management (save/load)
│   ├── intent.py      # Intent classification logic
│   ├── emotions.py    # Emotional state updates
│   ├── memory.py      # Memory compression logic
│   └── utils.py       # Helper functions
├── agent_state.json   # Persistent state storage
└── README.md          # Project documentation
```

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/kartikey071999/Cognitive-Chat-Control-System.git
    cd Cognitive-Chat-Control-System
    ```

2.  **Install dependencies**:
    Ensure you have Python installed. You may need `transformers` and `torch`.
    ```bash
    pip install transformers torch
    ```
    *Note: If you have a `uv.lock` or `pyproject.toml`, use the appropriate package manager (e.g., `uv sync`).*

## 🎮 Usage

Run the main application:

```bash
python main.py
```

Interact with the AI in the console. Type `exit` or `quit` to save the state and close the application.

## ⚙️ Configuration

You can modify `src/config.py` to change:
-   `MODEL_NAME`: The Hugging Face model to use.
-   `MAX_MEMORY_CHARS`: Threshold for triggering memory compression.
-   `EMOTIONS`: Initial emotional states.
-   `SYSTEM_RULES`: The system prompt guidelines.

## 🤝 Contributing

Feel free to submit issues or pull requests to improve the system!
