import os
import logging
from typing import List

from langchain.tools import BaseTool, Tool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader

# Import business extraction tools
try:
    from business_processor import get_business_extraction_tools
    BUSINESS_TOOLS_AVAILABLE = True
except ImportError:
    BUSINESS_TOOLS_AVAILABLE = False

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
    
    # Web Page Loader Tool using Playwright
    def fetch_webpage(url: str) -> str:
        """Fetch content from a webpage with full JavaScript rendering support."""
        try:
            # Create a loader for this specific URL
            loader = PlaywrightURLLoader(
                urls=[url],
                continue_on_failure=True,
                headless=True
            )
            docs = loader.load()
            if not docs:
                return f"No content could be loaded from {url}"
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.error(f"Error fetching webpage {url}: {str(e)}")
            return f"Error fetching webpage: {str(e)}"
    
    web_tool = Tool(
        name="FetchWebPage",
        description="Fetch and read content from a specific URL. Use this tool when you need to get detailed information from a webpage. Input should be a valid URL. This tool can handle JavaScript-rendered content.",
        func=fetch_webpage
    )
    tools.append(web_tool)
    logger.info("Playwright Web Page Loader tool added")
    
    # Add business extraction tools if available
    if BUSINESS_TOOLS_AVAILABLE:
        business_tools = get_business_extraction_tools()
        tools.extend(business_tools)
        logger.info(f"Added {len(business_tools)} business extraction tools")
    else:
        logger.warning("Business extraction tools not available")
    
    # Only using SerpAPI for search functionality
    if not os.environ.get("SERPAPI_API_KEY"):
        logger.warning("SERPAPI_API_KEY not found, search capabilities will not be available")
    
    return tools
