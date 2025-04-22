import asyncio
from typing import cast, Any, Literal, Dict, List, Optional
import json
import os

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
import google.generativeai as genai
from serpapi import GoogleSearch
from firecrawl import FirecrawlApp

from agent.configuration import Configuration
from agent.state import InputState, OutputState, OverallState
from agent.utils import deduplicate_and_format_sources, format_all_notes
from agent.prompts import (
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
)

# Configure rate limiter
rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,  # Controls the maximum burst size.
)

# Initialize Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Initialize FireCrawl
firecrawl_client = FirecrawlApp(api_key=os.environ.get("FIRECRAWL_API_KEY", ""))

# Initialize SerpAPI
serpapi_key = os.environ.get("SERPAPI_API_KEY", "")

# LLM Initialization
def get_llm(config: RunnableConfig):
    """Get the configured LLM."""
    configurable = Configuration.from_runnable_config(config)
    
    if configurable.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=configurable.gemini_model,
            temperature=0,
            convert_system_message_to_human=True,
            rate_limiter=rate_limiter
        )
    elif configurable.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-5-sonnet-latest",
            temperature=0,
            rate_limiter=rate_limiter
        )
    else:  # Default to gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            convert_system_message_to_human=True,
            rate_limiter=rate_limiter
        )

# Gemini direct model access for grounding search
def get_gemini_grounding_model():
    """Get a Gemini model with grounding search capabilities."""
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        },
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ],
        tools=[{"google_search": {}}],
    )

class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )


class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of field names that are missing or incomplete"
    )
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries
    llm = get_llm(config)

    # Generate search queries
    structured_llm = llm.with_structured_output(Queries)

    # Format system instructions
    person_str = f"Email: {state.person['email']}"
    if "name" in state.person:
        person_str += f" Name: {state.person['name']}"
    if "linkedin" in state.person:
        person_str += f" LinkedIn URL: {state.person['linkedin']}"
    if "role" in state.person:
        person_str += f" Role: {state.person['role']}"
    if "company" in state.person:
        person_str += f" Company: {state.person['company']}"

    query_instructions = QUERY_WRITER_PROMPT.format(
        person=person_str,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=max_search_queries,
    )

    # Generate queries
    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": "Please generate a list of search queries related to the schema that you want to populate.",
                },
            ]
        ),
    )

    # Queries
    query_list = [query for query in results.queries]
    return {"search_queries": query_list}


async def gemini_grounding_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Perform a search using Gemini's grounding capabilities."""
    try:
        model = get_gemini_grounding_model()
        search_prompt = f"""
        Search the web for information about the following person: {query}
        
        Present the results as a JSON list of the most relevant articles, 
        websites, or resources (max {max_results}). For each result, provide:
        
        1. The title of the page
        2. A comprehensive summary of the content (at least 200 words)
        3. The complete URL
        
        Format your response ONLY as a JSON list of objects, each with 'title', 'content', and 'url' keys.
        """
        
        response = model.generate_content(search_prompt)
        text = response.text
        
        # Extract JSON from response
        import re
        import json
        
        # Try to extract JSON from markdown code blocks or plain text
        json_match = re.search(r'```(?:json)?\s*(\[\s*\{.*?\}\s*\])\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Look for array pattern
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return []  # No valid JSON found
        
        results = json.loads(json_str)
        
        # Validate and format results
        formatted_results = []
        for item in results[:max_results]:
            if all(k in item for k in ["title", "content", "url"]):
                formatted_results.append({
                    "title": item["title"],
                    "content": item["content"],
                    "url": item["url"],
                })
        
        return formatted_results
    except Exception as e:
        print(f"Error in Gemini grounding search: {e}")
        return []


async def firecrawl_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Perform a search using FireCrawl."""
    try:
        search_results = firecrawl_client.search(
            query=query,
            search_options={"limit": max_results},
            page_options={"onlyMainContent": True}
        )
        
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "title": result.get("title", "Unknown Title"),
                "content": result.get("markdown", "No content available"),
                "url": result.get("url", "")
            })
            
        return formatted_results
    except Exception as e:
        print(f"Error in FireCrawl search: {e}")
        return []


async def serpapi_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Perform a search using SerpAPI."""
    try:
        if not serpapi_key:
            return []
            
        search = GoogleSearch({
            "q": query,
            "api_key": serpapi_key,
            "num": max_results
        })
        results = search.get_dict()
        
        formatted_results = []
        if "organic_results" in results:
            for result in results["organic_results"][:max_results]:
                # Get snippet or summary
                content = result.get("snippet", "")
                if not content and "rich_snippet" in result:
                    content = json.dumps(result["rich_snippet"])
                
                formatted_results.append({
                    "title": result.get("title", "Unknown Title"),
                    "content": content,
                    "url": result.get("link", "")
                })
                
        return formatted_results
    except Exception as e:
        print(f"Error in SerpAPI search: {e}")
        return []


async def combined_search(query: str, max_results: int = 3, config: RunnableConfig = None) -> List[Dict[str, Any]]:
    """Combined search using multiple search providers based on configuration."""
    configurable = Configuration.from_runnable_config(config)
    provider = configurable.search_provider
    enable_grounding = configurable.enable_grounding_search
    
    results = []
    
    # Primary search based on configuration
    if provider == "gemini" and enable_grounding:
        results = await gemini_grounding_search(query, max_results)
    elif provider == "firecrawl":
        results = await firecrawl_search(query, max_results)
    elif provider == "serpapi":
        results = await serpapi_search(query, max_results)
    elif provider == "combined":
        # Try all search providers and combine results
        search_tasks = []
        
        # Start with Gemini if grounding is enabled
        if enable_grounding:
            search_tasks.append(gemini_grounding_search(query, max_results))
        
        # Add FireCrawl search
        search_tasks.append(firecrawl_search(query, max_results))
        
        # Add SerpAPI if API key is available
        if serpapi_key:
            search_tasks.append(serpapi_search(query, max_results))
        
        # Execute all searches concurrently
        all_results = await asyncio.gather(*search_tasks)
        
        # Flatten and deduplicate by URL
        seen_urls = set()
        for result_set in all_results:
            for result in result_set:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append(result)
                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break
    
    # Default to Gemini grounding if no results and it wasn't already tried
    if not results and enable_grounding and provider != "gemini":
        results = await gemini_grounding_search(query, max_results)
    
    return results[:max_results]


async def research_person(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Execute a multi-step web search and information extraction process.

    This function performs the following steps:
    1. Executes concurrent web searches using the configured search provider
    2. Deduplicates and formats the search results
    """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_results = configurable.max_search_results
    llm = get_llm(config)

    # Web search
    search_tasks = []
    for query in state.search_queries:
        search_tasks.append(combined_search(query, max_search_results, config))

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    # Flatten search results into a format similar to what we expect
    formatted_search_results = []
    for result_set in search_docs:
        for result in result_set:
            formatted_search_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": 1.0,  # Default score
                "raw_content": result.get("content", "")  # Use content as raw_content
            })

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(
        [{"results": formatted_search_results}], 
        max_tokens_per_source=1000, 
        include_raw_content=True
    )

    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        people=state.person,
        user_notes=state.user_notes,
    )
    result = await llm.ainvoke(p)
    return {"completed_notes": [str(result.content)]}


def gather_notes_extract_schema(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""
    llm = get_llm(config)

    # Format all notes
    notes = format_all_notes(state.completed_notes)

    # Extract schema fields
    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), notes=notes
    )
    structured_llm = llm.with_structured_output(state.extraction_schema)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Produce a structured output from these notes.",
            },
        ]
    )
    return {"info": result}


def reflection(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find missing information."""
    llm = get_llm(config)
    structured_llm = llm.with_structured_output(ReflectionOutput)

    # Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema=json.dumps(state.extraction_schema, indent=2),
        info=state.info,
    )

    # Invoke
    result = cast(
        ReflectionOutput,
        structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Produce a structured reflection output."},
            ]
        ),
    )

    if result.is_satisfactory:
        return {"is_satisfactory": result.is_satisfactory}
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
        }


def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_person"]:  # type: ignore
    """Route the graph based on the reflection output."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # If we have satisfactory results, end the process
    if state.is_satisfactory:
        return END

    # If results aren't satisfactory but we haven't hit max steps, continue research
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_person"

    # If we've exceeded max steps, end even if not satisfactory
    return END


# Add nodes and edges
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)
builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
builder.add_node("generate_queries", generate_queries)
builder.add_node("research_person", research_person)
builder.add_node("reflection", reflection)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "research_person")
builder.add_edge("research_person", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)

# Compile
graph = builder.compile()
