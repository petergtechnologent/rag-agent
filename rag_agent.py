from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import os

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

load_dotenv()

# Read domain name and optional custom prompt from env
EXPERT_DOMAIN_NAME = os.getenv("EXPERT_DOMAIN_NAME", "generic_docs")
EXPERT_PROMPT = os.getenv("EXPERT_PROMPT", "You are a domain expert with knowledge from the crawled documentation.")

# Choose your default LLM model
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class RAGDeps:
    supabase: Client
    openai_client: AsyncOpenAI

# Build a dynamic system prompt
system_prompt = f"""
You are an expert on {EXPERT_DOMAIN_NAME}.
{EXPERT_PROMPT}

When a user asks a question, always consider using the retrieval tool to find relevant docs for {EXPERT_DOMAIN_NAME}.
If an answer is not found in the docs, be honest about it.
Provide references to the documents (URL or chunk title) you used, when applicable.
"""

rag_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=RAGDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536

@rag_agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[RAGDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks from site_pages where metadata->>'domain' = EXPERT_DOMAIN_NAME.
    """
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'domain': EXPERT_DOMAIN_NAME}
            }
        ).execute()

        if not result.data:
            return "No relevant documentation found."

        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
[Title]: {doc['title']}
[URL]: {doc['url']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)

        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@rag_agent.tool
async def list_documentation_pages(ctx: RunContext[RAGDeps]) -> List[str]:
    """
    Retrieve a list of all URLs for the current domain (metadata->>'domain' = EXPERT_DOMAIN_NAME).
    """
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>domain', EXPERT_DOMAIN_NAME) \
            .execute()
        if not result.data:
            return []
        return sorted(set(doc['url'] for doc in result.data))
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@rag_agent.tool
async def get_page_content(ctx: RunContext[RAGDeps], url: str) -> str:
    """
    Retrieve the full content for a page (all chunks) for the current domain.
    """
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>domain', EXPERT_DOMAIN_NAME) \
            .order('chunk_number') \
            .execute()

        if not result.data:
            return f"No content found for URL: {url}"

        page_title = result.data[0]['title']
        formatted_content = [f"# {page_title}\n"]
        for chunk in result.data:
            formatted_content.append(chunk['content'])
        return "\n\n".join(formatted_content)
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
