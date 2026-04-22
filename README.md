# 🎬 AutoStream AI Sales Agent

> A conversational AI agent for **AutoStream** — an automated video editing SaaS for content creators.
> Built with **LangGraph + Claude 3 Haiku**, featuring RAG-powered knowledge retrieval, intent detection, and automated lead capture.

---

## 📁 Project Structure

```
autostream-agent/
├── agent.py                    # Main LangGraph agent & CLI entrypoint
├── requirements.txt            # Python dependencies
├── knowledge_base/
│   └── autostream_kb.json      # Local knowledge base (pricing, policies, FAQs)
└── utils/
    ├── rag.py                  # TF-IDF based RAG retrieval pipeline
    └── tools.py                # Lead capture tool (mock API function)
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/autostream-agent.git
cd autostream-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your API Key

The agent uses **Claude 3 Haiku** via the Anthropic API.

```bash
export ANTHROPIC_API_KEY=your_api_key_here    # macOS / Linux
set ANTHROPIC_API_KEY=your_api_key_here       # Windows CMD
```

> Get your free API key at [console.anthropic.com](https://console.anthropic.com)

### 5. Run the Agent

```bash
python agent.py
```

You will see an interactive CLI prompt. Type your messages and press Enter.

---

## 💬 Example Conversation

```
You: Hi, what does AutoStream do?

Aria: Hi there! 👋 AutoStream is an AI-powered video editing SaaS platform
      designed specifically for content creators like you...

You: What's the pricing?

Aria: We have two plans:
      • Basic Plan – $29/month: 10 videos/month, 720p resolution, email support.
      • Pro Plan – $79/month: Unlimited videos, 4K, AI captions, 24/7 support.

You: I want to try the Pro plan for my YouTube channel!

Aria: That's awesome! I'd love to get you set up. Could I start with your full name?

You: Alex Johnson

Aria: Great, Alex! What's the best email address to reach you?

You: alex@example.com

Aria: Perfect! And you mentioned YouTube — is that your primary platform?

You: Yes, YouTube.

Aria: 
==================================================
  ✅  Lead captured successfully!
  Name     : Alex Johnson
  Email    : alex@example.com
  Platform : YouTube
==================================================

You're all set, Alex! Our team will reach out to alex@example.com shortly...
```

---

## 🏗️ Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it offers **explicit, inspectable state management** through a typed `StateGraph`. Every conversation turn flows through defined nodes and edges, making the agent's reasoning transparent and debuggable. Unlike AutoGen's multi-agent model, LangGraph's single-agent loop is more appropriate for a focused sales assistant that needs tight control over *when* a tool is triggered.

### How State is Managed

The agent maintains a `AgentState` TypedDict with two fields:

1. **`messages`** — A running list of all `HumanMessage`, `AIMessage`, and `ToolMessage` objects, accumulated via LangGraph's built-in `add_messages` reducer. This gives the LLM full conversation history across all turns.

2. **`lead_info`** — A dict that accumulates `name`, `email`, and `platform` as the conversation progresses. A heuristic extractor populates this from recent human messages (e.g., regex for emails), while the LLM itself manages the conversational flow of asking for each piece.

The graph has three nodes — `chat_node` (LLM reasoning), `tool_node` (tool execution), and a conditional router — forming a **loop** that only exits when no tool call is made. This ensures the agent can always respond after a tool execution.

### RAG Pipeline

Rather than a heavyweight vector store, a lightweight **TF-IDF similarity** engine retrieves the top-3 most relevant knowledge base chunks per user query. These chunks are injected into the system prompt dynamically, keeping the context window focused and reducing hallucination.

---

## 📱 WhatsApp Deployment via Webhooks

To integrate this agent with WhatsApp, the following architecture would be used:

### Overview

```
WhatsApp User
     │  (sends message)
     ▼
WhatsApp Business API (Meta)
     │  (HTTP POST webhook)
     ▼
FastAPI / Flask Backend  ←──── your server
     │
     ├── Parses incoming message & sender ID
     ├── Loads conversation state from Redis (keyed by sender phone number)
     ├── Invokes LangGraph agent with state + new message
     ├── Saves updated state back to Redis
     └── POSTs reply to WhatsApp Send Message API
```

### Step-by-step

1. **Register a Webhook** in the [Meta Developer Portal](https://developers.facebook.com/). Point it to `https://yourserver.com/webhook`.

2. **Verify the Webhook** — WhatsApp sends a `GET` request with a `hub.challenge` token. Your server must return it.

3. **Handle Incoming Messages** — On each `POST` to `/webhook`, extract:
   - `sender_phone` (unique user ID)
   - `message_body` (user text)

4. **Persist State** — Use **Redis** (or a database) to store `AgentState` keyed by `sender_phone`. This replaces the in-memory dict used in the CLI.

5. **Invoke the Agent** and send the AI response back via:
   ```
   POST https://graph.facebook.com/v18.0/{phone_number_id}/messages
   Authorization: Bearer {WHATSAPP_TOKEN}
   Body: { "to": sender_phone, "text": { "body": aria_response } }
   ```

6. **Lead Capture** — When `mock_lead_capture` fires, replace it with a real CRM API call (HubSpot, Salesforce, etc.).

### Key Considerations
- **Rate limits**: Meta enforces 80 messages/second per phone number.
- **Session TTL**: Set Redis keys to expire after 24–48 hours of inactivity.
- **Security**: Validate the `X-Hub-Signature-256` header on every webhook POST.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Agent Framework | LangGraph 0.2+ |
| LLM | Claude 3 Haiku (Anthropic) |
| RAG | TF-IDF (no external vector DB) |
| State | LangGraph `StateGraph` + TypedDict |
| Tool | LangChain `StructuredTool` |
| Knowledge Base | Local JSON file |

---

## 📊 Evaluation Criteria Coverage

| Criterion | Implementation |
|---|---|
| Agent reasoning & intent detection | System prompt with explicit intent categories; LLM decides flow |
| Correct use of RAG | TF-IDF retrieval injects relevant KB chunks per query |
| Clean state management | `AgentState` TypedDict; `add_messages` reducer; lead_info dict |
| Proper tool calling logic | Tool only fires after all 3 lead fields are collected |
| Code clarity & structure | Modular files: agent.py, rag.py, tools.py, KB JSON |
| Real-world deployability | WhatsApp webhook architecture documented above |

---

## 📄 License

MIT License — free to use and modify.
