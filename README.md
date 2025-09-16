# Warren Buffett AI Chatbot

An intelligent AI chatbot that simulates conversations with Warren Buffett using his actual quotes, interviews, and annual meeting transcripts. Built with OpenAI's GPT models, vector embeddings, and semantic search.

## Features

- **Warren Buffett Voice Simulation**: Attempts to simulate Warren Buffett's characteristic plain-spoken, Midwestern style using system prompts and real transcripts
- **Knowledge Base**: Built on real transcripts from Berkshire Hathaway annual meetings and interviews
- **Semantic Search**: Uses vector embeddings to find relevant Warren Buffett quotes
- **Multiple Interfaces**: REST API and Streamlit web interface
- **Investment Focus**: Specialized in investment advice and business philosophy

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Milvus Cloud account

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd warren-buffet-chatbot
   ```

2. **Install dependencies**

   ```bash
   pip install flask openai pandas requests beautifulsoup4
   transformers pymilvus streamlit langchain langchain-openai
   langchain-milvus python-dotenv
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   MILVUS_URI=your_milvus_cloud_uri
   MILVUS_TOKEN=your_milvus_token
   MILVUS_USER=your_milvus_user
   MILVUS_PASS=your_milvus_password
   ```

### Running the Application

#### Option 1: Streamlit Web Interface (Recommended for users)

```bash
streamlit run chat_app.py
```

- Opens at `http://localhost:8501`
- Modern chat interface with conversation memory

#### Option 2: Flask REST API (Recommended for developers)

```bash
python app.py
```

- Runs on `http://localhost:5000`
- RESTful endpoints for integration

#### Option 3: Test the API

```bash
python api_client.py
```

- Simple test script to verify API functionality

## API Endpoints

### Simple Question Endpoint

```http
POST /ask
Content-Type: application/json

{
  "role": "user",
  "content": "What do you think about investing in technology stocks?"
}
```

### Full Conversation Endpoint

```http
POST /chat_all
Content-Type: application/json

[
  {"role": "user", "content": "What do you think about bank failures?"},
  {"role": "assistant", "content": "Previous response..."},
  {"role": "user", "content": "How does that affect the economy?"}
]
```

### Example API Usage

```python
import requests
import json

# Simple question
response = requests.post('http://localhost:5000/ask',
                        json={'role': 'user', 'content': 'What is value investing?'})
print(response.json())

# Full conversation
messages = [
    {'role': 'user', 'content': 'What do you think about market timing?'},
    {'role': 'assistant', 'content': 'Previous response...'},
    {'role': 'user', 'content': 'What about dollar-cost averaging?'}
]
response = requests.post('http://localhost:5000/chat_all', json=messages)
print(response.json())
```

## Configuration

### Customizing the Chatbot

You can modify the system prompt in `wb_chat_completion.py`:

```python
self.initial_prompt = """You are [WARREN BUFFETT] and therefore need to answer the question in first-person.
You need to answer the question as truthfully as possible using the provided context text as a guidance.
If the answer is not contained within the context text, use best of your knowledge to answer.
If you are having problem coming up with answers, say "I don't know".
Provide longer explanation whenever possible.
"""
```

### Adjusting Search Parameters

- `top_n`: Number of relevant quotes to retrieve (default: 1)
- `distance_threshold`: Similarity threshold for context inclusion (default: 0.8)
- `temperature`: AI response creativity (default: 0.1 for consistency)

---
