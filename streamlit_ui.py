from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import subprocess

import streamlit as st
import json
import logfire
from supabase import create_client, Client
from openai import AsyncOpenAI

# Import pydantic_ai classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

# Import our domain-agnostic RAG agent
from rag_agent import rag_agent, RAGDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/UI."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    """
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)
    # You could handle ToolCallPart, ToolReturnPart, etc. as needed.

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = RAGDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    # Run the agent in a stream
    async with rag_agent.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Filter out user-prompt messages from the new messages
        filtered_messages = [
            msg for msg in result.new_messages()
            if not (hasattr(msg, 'parts') and any(part.part_kind == 'user-prompt' for part in msg.parts))
        ]
        st.session_state.messages.extend(filtered_messages)

        # Add the final assistant response to messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

async def main():
    st.title("RAG Agent")
    st.write("Ask questions about your custom domain or dataset. Provide domain details in the sidebar, then crawl, then chat.")

    # Sidebar config
    with st.sidebar:
        st.subheader("Domain Configuration")
        domain_name = st.text_input("Expert Domain Name", value=os.getenv("EXPERT_DOMAIN_NAME", "generic_docs"))
        sitemaps_input = st.text_area("Sitemap URLs (comma separated)", value=os.getenv("SITEMAP_URLS", ""))

        if st.button("Set Domain"):
            # Write environment variables (in a real app, you might manage these differently)
            os.environ["EXPERT_DOMAIN_NAME"] = domain_name
            os.environ["SITEMAP_URLS"] = sitemaps_input
            st.success(f"Set domain to '{domain_name}'. Sitemaps: {sitemaps_input}")

        if st.button("Crawl Now"):
            st.info("Crawling in progress; please wait...")
            # Run the crawler script as a subprocess
            proc = subprocess.run(["python", "crawl_any_docs.py"], capture_output=True, text=True)
            st.text(proc.stdout)
            if proc.returncode == 0:
                st.success("Crawl complete.")
            else:
                st.error(f"Error: {proc.stderr}")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("Ask a question about your domain docs...")

    if user_input:
        # We append a new request to the conversation
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Display user's question
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display streaming response
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main())
