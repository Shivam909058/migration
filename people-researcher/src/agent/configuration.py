import os
from dataclasses import dataclass, fields
from typing import Any, Optional, Literal

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the people researcher agent."""

    max_search_queries: int = 3  # Max search queries per person
    max_search_results: int = 3  # Max search results per query
    max_reflection_steps: int = 0  # Max reflection steps
    
    # New configuration options for search providers
    search_provider: Literal["gemini", "firecrawl", "serpapi", "combined"] = "combined"
    llm_provider: Literal["gemini", "anthropic", "openai"] = "gemini"
    gemini_model: str = "gemini-2.0-flash"  # Default Gemini model to use
    enable_grounding_search: bool = True  # Whether to use Gemini's grounding search

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
