"""
tools.py – Mock tool definitions for AutoStream agent.
Contains the lead-capture function and its LangChain StructuredTool wrapper.
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Mock backend function
# ─────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Simulates a CRM / backend API call to save a qualified lead.
    In production this would POST to a real CRM endpoint (HubSpot, Salesforce, etc.).
    """
    print(f"\n{'='*50}")
    print(f"  ✅  Lead captured successfully!")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"{'='*50}\n")
    return (
        f"Lead captured successfully! "
        f"Details saved — Name: {name}, Email: {email}, Platform: {platform}."
    )


# ─────────────────────────────────────────────
# Pydantic schema for type-safe tool calling
# ─────────────────────────────────────────────

class LeadCaptureInput(BaseModel):
    name: str = Field(..., description="Full name of the prospective customer")
    email: str = Field(..., description="Email address of the prospective customer")
    platform: str = Field(
        ...,
        description="Primary content platform (e.g. YouTube, Instagram, TikTok, etc.)"
    )


# ─────────────────────────────────────────────
# LangChain tool wrapper
# ─────────────────────────────────────────────

lead_capture_tool = StructuredTool.from_function(
    func=mock_lead_capture,
    name="lead_capture",
    description=(
        "Call this tool ONLY when you have collected the user's name, email address, "
        "and creator platform. Do NOT call this tool until all three pieces of "
        "information have been explicitly provided by the user."
    ),
    args_schema=LeadCaptureInput,
)
