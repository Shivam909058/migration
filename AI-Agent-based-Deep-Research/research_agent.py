import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.tools import Tool
import google.generativeai as genai
from firecrawl import FirecrawlApp
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    filename="research_agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize FireCrawl client
firecrawl_client = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

class ResearchResult(BaseModel):
    title: str
    content: str
    url: str

def gemini_grounded_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Perform a web search using Gemini's grounding with Google Search capability.
    
    Args:
        query: The search query.
        max_results: Maximum number of results to return.
        
    Returns:
        List of dictionaries containing search results.
    """
    try:
        # Configure Gemini model with Google search grounding
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 4096,
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
            tools=[{"google_search": {}}],
        )

        # Construct a system prompt that instructs the model to use Google Search
        system_prompt = f"""
        Search the web for information on: {query}
        
        Present the results as a list of the {max_results} most relevant articles, 
        websites, or resources. For each result, provide:
        
        1. The title of the page
        2. A comprehensive summary of the content (at least 200 words)
        3. The complete URL
        
        Format your response as a JSON list of objects, each with 'title', 'content', and 'url' keys.
        Ensure the summaries are detailed and informative.
        """

        # Generate content
        response = model.generate_content(
            system_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Parse the JSON response
        try:
            results = json.loads(response.text)
            # Validate structure
            for item in results:
                if not all(k in item for k in ['title', 'content', 'url']):
                    raise ValueError("Missing required fields in response")
            return results[:max_results]
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse Gemini response: {e}")
            logging.error(f"Raw response: {response.text}")
            return []
            
    except Exception as e:
        logging.error(f"Gemini search error: {str(e)}")
        return []

def firecrawl_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a web search using FireCrawl.
    
    Args:
        query: The search query.
        max_results: Maximum number of results to return.
        
    Returns:
        List of dictionaries containing search results.
    """
    try:
        # Search using FireCrawl
        search_results = firecrawl_client.search(
            query=query, 
            search_options={"limit": max_results},
            page_options={"onlyMainContent": True}
        )
        
        # Format the results
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "title": result.get("title", "Unknown Title"),
                "content": result.get("markdown", "No content available"),
                "url": result.get("url", "")
            })
            
        return formatted_results
    except Exception as e:
        logging.error(f"FireCrawl search error: {str(e)}")
        return []

def firecrawl_scrape_page(url: str) -> Optional[Dict[str, Any]]:
    """
    Scrape detailed content from a specific URL using FireCrawl.
    
    Args:
        url: The URL to scrape.
        
    Returns:
        Dictionary containing the scraped content or None if scraping fails.
    """
    try:
        # Scrape the URL
        scrape_result = firecrawl_client.scrape_url(
            url=url,
            formats=["markdown"],
            page_options={"onlyMainContent": True}
        )
        
        # Check if scraping was successful
        if not scrape_result or not scrape_result[0].get("markdown"):
            return None
            
        # Return formatted result
        return {
            "title": scrape_result[0].get("title", "Unknown Title"),
            "content": scrape_result[0].get("markdown", "No content available"),
            "url": url
        }
    except Exception as e:
        logging.error(f"FireCrawl scrape error for {url}: {str(e)}")
        return None

def research_web(query: str, deep_research: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch data from the web using Gemini's grounding search and FireCrawl.
    
    Args:
        query: The search query.
        deep_research: Whether to perform deep research (more results).
        
    Returns:
        List of dictionaries containing research data.
    """
    try:
        # Adjust max_results based on deep_research mode
        max_results = 20 if deep_research else 5
        data = []
        url_set = set()

        # Initial Gemini grounded search
        grounded_results = gemini_grounded_search(query, max_results=max_results)
        for item in grounded_results:
            if item["url"] not in url_set:
                data.append(item)
                url_set.add(item["url"])

        # Supplement with FireCrawl search results
        if deep_research and len(data) < max_results:
            firecrawl_results = firecrawl_search(query, max_results=max_results)
            for item in firecrawl_results:
                if item["url"] not in url_set:
                    data.append(item)
                    url_set.add(item["url"])

        # If deep research mode and fewer than target results, try additional queries
        if deep_research and len(data) < max_results:
            logging.info(f"Initial queries returned {len(data)} results, attempting additional queries...")
            # List of variant queries to broaden the search
            variant_queries = [
                f"{query} overview OR review OR advancements OR trends",
                f"{query} recent developments OR innovations OR breakthroughs",
                f"{query} applications OR use cases OR impact"
            ]
            for variant_query in variant_queries:
                if len(data) >= max_results:
                    break
                additional_results = gemini_grounded_search(variant_query, max_results=10)
                for item in additional_results:
                    if item["url"] not in url_set:
                        data.append(item)
                        url_set.add(item["url"])

        # For each high-priority URL, try to scrape additional detailed content
        if deep_research:
            for i, item in enumerate(data[:5]):  # Focus on top 5 results
                detailed_content = firecrawl_scrape_page(item["url"])
                if detailed_content and len(detailed_content["content"]) > len(item["content"]):
                    data[i] = detailed_content

        # Limit results to avoid overwhelming the model
        max_total = 30 if deep_research else 10
        data = data[:max_total]

        # Save results to file
        with open("research_data.json", "w") as f:
            json.dump(data, f, indent=2)
            
        logging.info(f"Fetched {len(data)} research items")
        return data
    except Exception as e:
        error_msg = f"Research failed: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)

research_tool = Tool(
    name="WebResearch",
    func=lambda query, deep_research=False: research_web(query, deep_research),
    description="Fetches data from the web using Gemini grounding search and FireCrawl. Supports deep research mode with more results."
)