from langgraph.graph import Graph
from research_agent import research_tool
from draft_agent import draft_tool
from joblib import Memory
import logging

# Set up logging
logging.basicConfig(
    filename="research_agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize caching with joblib
memory = Memory("cache", verbose=0)
memory.clear()  # Clear the cache

# Define the research node to update the state
@memory.cache
def fetch_research_data(query: str, deep_research: bool = False) -> list:
    """Fetch research data using the research tool with caching."""
    logging.info(f"Fetching research data for query: {query}, deep_research: {deep_research}")
    result = research_tool.run(query, deep_research=deep_research)
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and "error" in result[0]:
        error_msg = f"Research failed: {result[0]['error']}"
        logging.error(error_msg)
        raise Exception(error_msg)
    return result

def research_node(state):
    """Fetch research data and update the state."""
    query = state["query"]
    deep_research = state.get("deep_research", False)
    try:
        logging.info(f"Running research node with query: {query}, deep_research: {deep_research}")
        research_data = fetch_research_data(query, deep_research)
        state["research"] = research_data
        logging.info(f"Research node completed successfully with {len(research_data)} results")
        return state
    except Exception as e:
        logging.error(f"Research node failed: {str(e)}")
        state["error"] = str(e)
        return state

# Define the draft node to use research data and update the state
def draft_node(state):
    """Draft a summary using research data and update the state."""
    if "error" in state:
        logging.error(f"Draft node skipped due to previous error: {state['error']}")
        return state
        
    research_data = state["research"]
    if not isinstance(research_data, list):
        error_msg = "Research data is not in the expected format (list required)"
        logging.error(error_msg)
        state["error"] = error_msg
        return state
    
    # Extract all parameters from state
    deep_research = state.get("deep_research", False)
    target_word_count = state.get("target_word_count", 1000)
    writing_style = state.get("writing_style", "academic")
    citation_format = state.get("citation_format", "APA")
    language = state.get("language", "english")
    
    try:
        logging.info(f"Running draft node with {len(research_data)} research items")
        result = draft_tool.invoke({
            "data": research_data,
            "deep_research": deep_research,
            "target_word_count": target_word_count,
            "writing_style": writing_style,
            "citation_format": citation_format,
            "language": language,
            "retries": 3,
            "delay": 5
        })
        if "Error drafting response" in result:
            error_msg = f"Draft node failed: {result}"
            logging.error(error_msg)
            state["error"] = error_msg
            return state
            
        state["draft"] = result
        logging.info("Draft node completed successfully")
        return state
    except Exception as e:
        error_msg = f"Draft node failed with exception: {str(e)}"
        logging.error(error_msg)
        state["error"] = error_msg
        return state

# Define error handling
def handle_error(state):
    """Handle errors in the workflow."""
    if "error" in state:
        logging.error(f"Workflow error handled: {state['error']}")
        return "end"
    return "next_step"

# Initialize the graph
workflow = Graph()

# Add nodes to the workflow
workflow.add_node("research", research_node)
workflow.add_node("draft", draft_node)

# Define edges
workflow.add_edge("research", "draft")

# Set entry and finish points
workflow.set_entry_point("research")
workflow.set_finish_point("draft")

# Compile the workflow
app = workflow.compile()

# Function to run the research system
def run_research(query: str, deep_research: bool = False, target_word_count: int = 1000, writing_style: str = "academic", citation_format: str = "APA", language: str = "english") -> tuple:
    """Run the research workflow and return results."""
    input_dict = {
        "query": query,
        "deep_research": deep_research,
        "target_word_count": target_word_count,
        "writing_style": writing_style,
        "citation_format": citation_format,
        "language": language
    }
    
    try:
        logging.info(f"Starting research workflow with input: {input_dict}")
        result = app.invoke(input_dict)
        # Ensure result is a dictionary and extract outputs
        if not isinstance(result, dict):
            error_msg = f"Workflow returned unexpected type: {type(result)}"
            logging.error(error_msg)
            return [], f"Workflow failed: {error_msg}"
            
        if "error" in result:
            logging.error(f"Workflow completed with error: {result['error']}")
            return [], f"Workflow failed: {result['error']}"
            
        research_data = result.get("research", [])
        draft_response = result.get("draft", "Error: Draft not generated")
        logging.info("Research workflow completed successfully")
        return research_data, draft_response
    except Exception as e:
        error_msg = f"Workflow failed with exception: {str(e)}"
        logging.error(error_msg)
        # Return a tuple with empty list and error message instead of raising
        return [], error_msg

# Example usage
if __name__ == "__main__":
    query = "latest advancements in quantum computing"
    deep_research = True
    target_word_count = 2000
    
    print(f"Researching: {query} (Deep research: {deep_research})")
    research_data, response = run_research(
        query, 
        deep_research=deep_research,
        target_word_count=target_word_count
    )
    
    print(f"Found {len(research_data)} research items")
    print("Response generated successfully!")
    
    # Print a small preview
    preview_lines = response.split('\n')[:10]
    print("\nPreview of the response:")
    print("\n".join(preview_lines))
    print("...")
