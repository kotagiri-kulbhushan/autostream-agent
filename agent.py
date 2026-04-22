import os
import json
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from utils.rag import retrieve, get_full_context
from utils.tools import lead_capture_tool

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# State definition
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    lead_info: dict          # accumulates name / email / platform


# ─────────────────────────────────────────────
# System prompt factory
# ─────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are Aria, the friendly and knowledgeable AI sales assistant for AutoStream — \
a SaaS platform that provides automated, AI-powered video editing tools for content creators.

## Your Responsibilities
1. **Greet** users warmly and answer general questions.
2. **Answer product / pricing questions** accurately using ONLY the knowledge base provided below.
3. **Identify high-intent users** — someone who says they want to sign up, try the product, \
get started, or mentions being ready to purchase is a high-intent lead.
4. **Capture lead details** — when a user shows high intent, politely collect:
   - Their full name
   - Email address
   - Primary creator platform (YouTube, Instagram, TikTok, etc.)
   Collect ONE piece of info at a time; do not bombard the user.
5. **Call the `lead_capture` tool** ONLY after you have ALL THREE: name, email, AND platform.
   Never call the tool prematurely.

## Tone
- Friendly, helpful, and concise.
- Never make up information not in the knowledge base.
- If unsure, say "I don't have that information right now, but our support team can help!"

## Knowledge Base
{kb_context}

## Current Lead Info Collected
{lead_info}
"""


def build_system_prompt(query: str, lead_info: dict) -> str:
    kb_context = retrieve(query, top_k=3)
    lead_str = json.dumps(lead_info, indent=2) if lead_info else "Nothing collected yet."
    return SYSTEM_TEMPLATE.format(kb_context=kb_context, lead_info=lead_str)


# ─────────────────────────────────────────────
# LLM setup
# ─────────────────────────────────────────────

def get_llm():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Check your .env file."
        )

    llm = ChatOpenAI(
        model="gpt-5.4-mini",   # fast + cheap
        api_key=api_key,
        temperature=0.3,
        max_tokens=1024,
    )

    return llm.bind_tools([lead_capture_tool])

# ─────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────

def chat_node(state: AgentState) -> AgentState:
    """Main reasoning node — calls the LLM with current state."""
    llm = get_llm()

    # Build a dynamic system message based on the latest user query
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    system_msg = SystemMessage(content=build_system_prompt(last_human, state["lead_info"]))

    response: AIMessage = llm.invoke([system_msg] + state["messages"])

    # If the AI is asking for lead info, try to extract it from recent messages
    lead_info = dict(state["lead_info"])
    _extract_lead_info(state["messages"], lead_info)

    return {"messages": [response], "lead_info": lead_info}


def tool_node(state: AgentState) -> AgentState:
    """Executes any tool calls requested by the LLM."""
    last_ai: AIMessage = state["messages"][-1]
    tool_messages = []

    for tool_call in last_ai.tool_calls:
        if tool_call["name"] == "lead_capture":
            result = lead_capture_tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

    return {"messages": tool_messages, "lead_info": state["lead_info"]}


def _extract_lead_info(messages: list, lead_info: dict) -> None:
    """
    Heuristic: scan the last few human messages for name / email / platform clues.
    The LLM itself will do the heavy lifting via prompting; this is a lightweight backup.
    """
    import re

    email_re = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    platform_keywords = ["youtube", "instagram", "tiktok", "twitter", "facebook", "linkedin", "x.com"]

    recent_human = [
        m.content for m in messages[-6:] if isinstance(m, HumanMessage)
    ]

    for text in recent_human:
        # Email
        if "email" not in lead_info:
            match = email_re.search(text)
            if match:
                lead_info["email"] = match.group()

        # Platform
        if "platform" not in lead_info:
            lower = text.lower()
            for kw in platform_keywords:
                if kw in lower:
                    lead_info["platform"] = kw.capitalize()
                    break


# ─────────────────────────────────────────────
# Routing logic
# ─────────────────────────────────────────────

def should_use_tool(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool"
    return "end"


# ─────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("chat_node", chat_node)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "chat_node")
    builder.add_conditional_edges(
        "chat_node",
        should_use_tool,
        {"tool": "tool_node", "end": END},
    )
    builder.add_edge("tool_node", "chat_node")  # loop back after tool execution

    return builder.compile()


# ─────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────

def run_cli():
    print("\n" + "="*60)
    print("  🎬  AutoStream AI Assistant  (type 'quit' to exit)")
    print("="*60 + "\n")

    graph = build_graph()
    state: AgentState = {"messages": [], "lead_info": {}}

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("\nAria: Thanks for chatting! Have a great day 👋\n")
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = graph.invoke(state)

        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai and last_ai.content:
            print(f"\nAria: {last_ai.content}\n")


if __name__ == "__main__":
    run_cli()
