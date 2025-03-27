import os
import logging
from typing import List

from langchain.tools import BaseTool, Tool
from langchain_community.tools import SerpAPIWrapper
from langchain_community.document_loaders import AsyncWebPageLoader
from langchain_community.tools.tavily_search import TavilySearchResults

# Configure logging
logger = logging.getLogger(__name__)

def get_tools() -> List[BaseTool]:
    """Return a list of tools to be used by the agent."""
    tools = []
    
    # SerpAPI Search Tool
    try:
        serpapi_key = os.environ.get("SERPAPI_API_KEY")
        if serpapi_key:
            search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
            search_tool = Tool(
                name="Search",
                description="Useful to search the internet for current information. Use this tool when you need to find information about events, people, or any other topic.",
                func=search.run
            )
            tools.append(search_tool)
            logger.info("SerpAPI Search tool added")
        else:
            logger.warning("SERPAPI_API_KEY not found, search tool will not be available")
    except Exception as e:
        logger.error(f"Error setting up SerpAPI tool: {str(e)}")
    
    # Async Web Page Loader Tool
    def fetch_webpage(url: str) -> str:
        """Fetch content from a webpage asynchronously."""
        try:
            loader = AsyncWebPageLoader(url)
            docs = loader.load()
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.error(f"Error fetching webpage {url}: {str(e)}")
            return f"Error fetching webpage: {str(e)}"
    
    web_tool = Tool(
        name="FetchWebPage",
        description="Fetch and read content from a specific URL. Use this tool when you need to get detailed information from a webpage. Input should be a valid URL.",
        func=fetch_webpage
    )
    tools.append(web_tool)
    logger.info("Async Web Page Loader tool added")
    
    # Fallback to Tavily Search if SerpAPI key is not available
    if not os.environ.get("SERPAPI_API_KEY"):
        try:
            tavily_api_key = os.environ.get("TAVILY_API_KEY")
            if tavily_api_key:
                tavily_tool = TavilySearchResults(api_key=tavily_api_key)
                tools.append(tavily_tool)
                logger.info("Tavily Search tool added as fallback")
            else:
                logger.warning("Neither SERPAPI_API_KEY nor TAVILY_API_KEY found, search capabilities limited")
        except Exception as e:
            logger.error(f"Error setting up Tavily search tool: {str(e)}")
    
    return tools
