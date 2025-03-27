"""Workflow for finding owner addresses based on identified business owners."""

import os
import time
import asyncio
import json
from typing import Dict, List, Any, Optional, TypedDict, cast, Callable
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.utilities import SerpAPIWrapper
from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from core.utils.logging import logger, log_substep
from core.utils.cache import SQLiteCache
from core.configs.llm_configs import build_llm

from core.utils.webpage import js_aware_loader, clean_html_content, is_people_search_site
from core.utils.crawler import recursive_crawl
from core.prompts.lead_enrichment.address_search import (
    OwnerAddress,
    AddressSearchQuery,
    WebpageScoreAddress,
    get_address_query_generation_prompt,
    get_address_extraction_prompt,
    get_address_synthesis_prompt,
    get_webpage_scoring_prompt,
)

# Rich console for better formatting
console = Console()

# Maximum number of concurrent operations
MAX_CONCURRENT_SEARCHES = 5
MAX_CONCURRENT_EXTRACTIONS = 3
MAX_CONCURRENT_SCORINGS = 5

# More aggressive search and extraction settings
MAX_SEARCH_RESULTS = 50  # Increased from 30
MAX_RETRIES = 3
MIN_CONFIDENCE_THRESHOLD = 4  # Lower threshold to accept more potential addresses

# Initialize cache
cache = SQLiteCache()

# Define the state for our graph
class AddressSearchState(TypedDict):
    """State for address search workflow."""
    owner_name: str
    business_name: str
    business_state: str
    business_zip: str
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    webpage_scores: List[Dict[str, Any]]  # List of scored webpages for address search
    webpage_contents: List[Dict[str, Any]]
    extracted_addresses: List[Dict[str, Any]]
    final_address: Optional[Dict[str, Any]]
    errors: List[str]

class AddressExtractionResult(BaseModel):
    """Result of address extraction from a single webpage."""
    addresses: List[Dict[str, Any]] = Field(description="List of potential addresses found")
    owner_matches: List[str] = Field(description="Names that seem to match the owner name")
    is_residential: bool = Field(description="Whether the addresses appear to be residential")
    reasoning: str = Field(description="Reasoning for confidence in each address")

# Initialize the LLM
def build_llm():
    """Get the LLM for the workflow."""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

# Node 1: Generate search queries for address
async def generate_address_queries(state: AddressSearchState) -> AddressSearchState:
    """Generate search queries for finding the owner's address."""
    console.print(Panel(f"[bold blue]Generating Address Search Queries for[/bold blue] [yellow]{state['owner_name']}[/yellow]"))
    
    start_time = time.time()
    
    # Check cache first
    cached_queries = cache.get_address_queries(
        state["owner_name"],
        state["business_name"],
        state["business_state"],
        state["business_zip"]
    )
    
    if cached_queries:
        logger.info(f"Using {len(cached_queries)} cached address search queries for {state['owner_name']}")
        duration = time.time() - start_time
        
        for i, query in enumerate(cached_queries, 1):
            log_substep(f"Query {i}: {query}")
            
        logger.debug(f"Retrieved cached address queries in {duration:.2f} seconds")
        
        return {
            **state,
            "search_queries": cached_queries
        }
    
    # Generate new queries if not in cache
    llm = build_llm()
    
    # Initialize the Pydantic output parser
    parser = PydanticOutputParser(pydantic_object=AddressSearchQuery)
    format_instructions = parser.get_format_instructions()
    
    # Get the prompt
    prompt = get_address_query_generation_prompt()
    
    logger.info(f"Generating address search queries for {state['owner_name']}")
    
    # Run the prompt
    chain = prompt.partial(format_instructions=format_instructions) | llm | parser
    
    response = await chain.ainvoke({
        "person_name": state["owner_name"],
        "business_name": state["business_name"],
        "business_state": state["business_state"],
        "business_zip": state["business_zip"]
    })
    
    # Log generated queries
    logger.info(f"Generated {len(response.queries)} address search queries for {state['owner_name']}")
    for i, query in enumerate(response.queries, 1):
        log_substep(f"Query {i}: {query}")
    
    duration = time.time() - start_time
    logger.debug(f"Address search query generation took {duration:.2f} seconds")
    
    # Cache the generated queries
    cache.cache_address_queries(
        state["owner_name"],
        state["business_name"],
        state["business_state"],
        state["business_zip"],
        response.queries
    )
    
    # Update state with generated queries
    return {
        **state,
        "search_queries": response.queries
    }

# Function to execute a single search query
async def execute_single_address_query(query: str, search: SerpAPIWrapper) -> Dict[str, Any]:
    """Execute a single address search query and return the results."""
    start_time = time.time()
    
    # Check cache first
    cached_results = cache.get_search_results(query)
    if cached_results:
        query_time = time.time() - start_time
        results_count = len(cached_results)
        log_substep(f"Using {results_count} cached results for query: {query} ({query_time:.2f}s)")
        
        return {
            "query": query,
            "results": cached_results,
            "count": results_count,
            "time": query_time,
            "cached": True
        }
    
    # Execute search if not in cache
    try:
        # Execute search
        search_result = await search.aresults(query)
        
        # Calculate execution time
        query_time = time.time() - start_time
        
        # Extract organic results and store them
        results = []
        results_count = 0
        if "organic_results" in search_result:
            results_count = len(search_result["organic_results"])
            
            for result in search_result["organic_results"]:
                results.append({
                    "query": query,
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                })
        
        log_substep(f"Found {results_count} results in {query_time:.2f}s")
        
        # Cache the results
        if results:
            cache.cache_search_results(query, results)
        
        return {
            "query": query,
            "results": results,
            "count": results_count,
            "time": query_time,
            "cached": False
        }
    except Exception as e:
        logger.error(f"Error executing address search query '{query}': {str(e)}")
        return {
            "query": query,
            "results": [],
            "count": 0,
            "time": time.time() - start_time,
            "error": str(e),
            "cached": False
        }

# Node 2: Execute address search queries
async def execute_address_searches(state: AddressSearchState) -> AddressSearchState:
    """Execute search queries to find owner's address."""
    console.print(Panel(f"[bold blue]Searching for Address of[/bold blue] [yellow]{state['owner_name']}[/yellow]"))
    
    # Check if SERPAPI_API_KEY is set
    if not os.environ.get("SERPAPI_API_KEY"):
        logger.error("SERPAPI_API_KEY not set in environment")
        raise ValueError("SERPAPI_API_KEY environment variable is required but not set")
    
    try:
        search = SerpAPIWrapper()
    except ImportError as e:
        logger.error(f"Error initializing SerpAPI: {str(e)}")
        raise
    
    all_search_results = []
    search_stats = {
        "total_queries": len(state["search_queries"]),
        "total_results": 0,
        "query_times": [],
        "cached_count": 0
    }
    
    logger.info(f"Executing {search_stats['total_queries']} address search queries in parallel (max {MAX_CONCURRENT_SEARCHES} concurrent)")
    start_time = time.time()
    
    # Execute search queries in parallel using asyncio.gather with semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)
    
    async def wrapped_search(i, query):
        #async with semaphore:
            logger.info(f"Executing address search query ({i}/{search_stats['total_queries']}): {query}")
            return await execute_single_address_query(query, search)
    
    # Run queries in parallel with controlled concurrency
    search_tasks = [wrapped_search(i, query) for i, query in enumerate(state["search_queries"], 1)]
    search_results = await asyncio.gather(*search_tasks)
    
    # Process results
    for result in search_results:
        if "error" not in result:
            search_stats["query_times"].append(result["time"])
            search_stats["total_results"] += result["count"]
            
            if result.get("cached", False):
                search_stats["cached_count"] += 1
                
            all_search_results.extend(result["results"])
    
    # Log search statistics
    total_duration = time.time() - start_time
    avg_time = sum(search_stats["query_times"]) / len(search_stats["query_times"]) if search_stats["query_times"] else 0
    logger.info(f"Address search completed in {total_duration:.2f}s: {search_stats['total_results']} total results from {search_stats['total_queries']} queries ({search_stats['cached_count']} cached)")
    logger.debug(f"Average query time: {avg_time:.2f}s")
    
    # Display search summary
    console.print(f"[green]✓[/green] Found [bold]{search_stats['total_results']}[/bold] address search results in [bold]{total_duration:.2f}s[/bold] ({search_stats['cached_count']} from cache)")
    
    # Update state with search results
    return {
        **state,
        "search_results": all_search_results
    }

# Function to extract content from a single webpage
async def extract_single_address_webpage(i: int, total: int, result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract content from a single webpage for address search."""
    start_time = time.time()
    
    # Ensure we have a valid URL
    url = result.get('link', '')
    if not url:  # Try alternative keys if 'link' is empty
        url = result.get('url', result.get('displayLink', ''))
    
    # Make sure URL is a string and not None
    url = str(url) if url is not None else ''
    
    # If no valid URL, raise error - we cannot proceed without a URL
    if not url:
        raise ValueError("No URL found in search result")
    
    # Format URL for display
    display_url = url[:50] + "..." if len(url) > 50 else url
    
    logger.info(f"Extracting content from address webpage {i}/{total}: {url}")
    
    # Check cache first
    cached_content = cache.get_webpage_content(url)
    if cached_content:
        duration = time.time() - start_time
        content_length = len(cached_content["content"])
        logger.debug(f"Using cached content for {url}: {content_length} characters")
        log_substep(f"Using cached content ({content_length} chars) from {display_url}")
        
        return {
            "url": url,
            "title": cached_content.get("title", result.get("title", "")),
            "content": cached_content["content"],
            "duration": duration,
            "cached": True
        }
    
    # Extract new content if not in cache
    try:
        # Load webpage using JavaScript-aware loader
        logger.debug(f"Loading webpage: {url}")
        documents = await js_aware_loader(
            url=url,
            timeout=60,  # Use a reasonable timeout
            max_retries=2,  # Allow for retries
            query=result.get("query", None)  # Pass the original query if available
        )
        
        # Try recursive crawling if initial load fails
        if not documents or len(documents) == 0 or not documents[0].page_content.strip():
            logger.debug(f"Initial load failed or empty, trying recursive crawl for: {url}")
            documents = await recursive_crawl(
                url=url,
                max_depth=1
            )
        
        # Extract content from documents
        if documents and len(documents) > 0 and documents[0].page_content.strip():
            # Clean and process the HTML content
            raw_content = documents[0].page_content
            content = clean_html_content(raw_content)
            
            # Make sure we have actual content
            if not content or len(content.strip()) < 100:
                logger.warning(f"Extracted content from {url} is too short or empty, using raw content")
                content = raw_content
            
            content_length = len(content)
            logger.debug(f"Extracted {content_length} characters from {url}")
            
            duration = time.time() - start_time
            log_substep(f"Extracted {content_length} chars in {duration:.2f}s from {display_url}")
            
            # Cache the extracted content
            title = documents[0].metadata.get("title", "")
            cache.cache_webpage_content(url, content, title)
            
            return {
                "url": url,
                "title": title,
                "content": content,
                "duration": duration,
                "cached": False
            }
        else:
            # If we couldn't get content, fail loudly
            raise ValueError(f"No content could be extracted from {url}")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error extracting content from {url}: {str(e)}")
        log_substep(f"[red]Error extracting from {display_url}: {str(e)}[/red]")
        
        # Raise the error instead of returning a fallback
        raise ValueError(f"Error extracting webpage content: {str(e)}")

# Function to score a single webpage for address relevance
async def score_single_address_webpage(
    i: int, 
    total: int, 
    result: Dict[str, Any], 
    chain: Any, 
    owner_name: str,
    business_name: str, 
    business_state: str, 
    business_zip: str
) -> Dict[str, Any]:
    """Score a single webpage for address relevance."""
    item_start = time.time()
    url = result.get('link', '')
    if not url:  # Ensure we have a valid URL
        url = result.get('url', result.get('displayLink', 'unknown-url'))
    
    # Make sure url is a string
    url = str(url) if url is not None else 'unknown-url'
    
    # Format URL for display
    display_url = url[:50] + "..." if len(url) > 50 else url
    
    # Check cache first
    cache_key = f"address_webpage_score_{url}_{owner_name}"
    cached_score = cache.get_search_results(cache_key)
    
    if cached_score:
        item_duration = time.time() - item_start
        address_indicator = "✓" if cached_score.get("address_likely_present", False) else "✗"
        log_substep(f"Using cached score for {display_url} - Relevance: {cached_score['relevance_score']}/10, Address likely: {address_indicator} ({item_duration:.2f}s)")
        return cached_score
    
    try:
        logger.debug(f"Scoring webpage {i}/{total} for address relevance: {url}")
        
        # Get content for scoring - use snippet if available
        webpage_content = result.get("snippet", "")
        if not webpage_content:
            webpage_content = result.get("title", "")
            
        # Invoke the chain properly using the LCEL approach with ALL required parameters
        response = await chain.ainvoke({
            "person_name": owner_name,
            "business_name": business_name,
            "business_state": business_state,
            "business_zip": business_zip,
            "webpage_url": url,
            "webpage_content": webpage_content
        })
        
        # Since we're using a PydanticOutputParser, response is already a WebpageScoreAddress object
        item_duration = time.time() - item_start
        address_indicator = "✓" if response.address_likely_present else "✗"
        log_substep(f"Scored {display_url} - Relevance: {response.relevance_score}/10, Address likely: {address_indicator} ({item_duration:.2f}s)")
        
        # Create and cache the score
        score_result = {
            "url": url,
            "title": result.get("title", ""),
            "relevance_score": response.relevance_score,
            "address_likely_present": response.address_likely_present,
            "reasoning": response.reasoning,
            "duration": item_duration
        }
        
        cache.cache_search_results(cache_key, score_result)
        
        return score_result
    except Exception as e:
        item_duration = time.time() - item_start
        logger.error(f"Error scoring webpage {url} for address: {str(e)}")
        
        # Raise the error instead of returning a fallback
        raise ValueError(f"Error scoring webpage for address relevance: {str(e)}")

# Add a new node between execute_address_searches and extract_address_webpage_content
async def score_address_webpages(state: AddressSearchState) -> AddressSearchState:
    """Score webpages for address relevance before content extraction."""
    console.print(Panel(f"[bold blue]Scoring Webpage Relevance for Address of[/bold blue] [yellow]{state['owner_name']}[/yellow]"))
    
    if not state["search_results"]:
        error_msg = f"No search results to score for {state['owner_name']}"
        logger.warning(error_msg)
        raise ValueError(f"No search results available to score for {state['owner_name']}")
    
    start_time = time.time()
    
    # Initialize the PydanticOutputParser with the WebpageScoreAddress model
    parser = PydanticOutputParser(pydantic_object=WebpageScoreAddress)
    format_instructions = parser.get_format_instructions()
    
    # Get the webpage scoring prompt
    prompt = get_webpage_scoring_prompt()
    
    # Create a properly structured LCEL chain
    llm = build_llm()
    chain = prompt.partial(format_instructions=format_instructions) | llm | parser
    
    logger.info(f"Scoring {len(state['search_results'])} webpages for address relevance")
    
    # Score webpages in parallel with concurrency limit
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCORINGS)
    
    async def wrapped_score(i, result):
        #async with semaphore:
            return await score_single_address_webpage(
                i,
                len(state['search_results']),
                result,
                chain, 
                state['owner_name'],
                state['business_name'],
                state['business_state'],
                state['business_zip']
            )
    
    # Run scoring in parallel with controlled concurrency
    score_tasks = [wrapped_score(i, result) for i, result in enumerate(state["search_results"], 1)]
    all_scores = await asyncio.gather(*score_tasks)
    
    # Filter and sort scores by relevance_score (highest first)
    webpage_scores = sorted(
        [score for score in all_scores if score.get("relevance_score", 0) > 0],
        key=lambda x: x.get("relevance_score", 0),
        reverse=True
    )
    
    # Take the top results with highest scores
    top_webpages = webpage_scores[:MAX_SEARCH_RESULTS]
    
    # Log results
    total_duration = time.time() - start_time
    logger.info(f"Address webpage scoring completed in {total_duration:.2f}s: selected {len(top_webpages)} out of {len(webpage_scores)} scored webpages")
    
    # Display results summary
    console.print(f"[green]✓[/green] Selected [bold]{len(top_webpages)}[/bold] most relevant webpages for address in [bold]{total_duration:.2f}s[/bold]")
    
    # If no relevant webpages, raise error - no fallbacks
    if not top_webpages:
        error_msg = f"No relevant webpages found for {state['owner_name']}'s address"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Update state with webpage scores
    return {
        **state,
        "webpage_scores": top_webpages
    }

# Node 3: Extract content from search results
async def extract_address_webpage_content(state: AddressSearchState) -> AddressSearchState:
    """Extract content from address search results."""
    console.print(Panel(f"[bold blue]Extracting Content from Address Search Results for[/bold blue] [yellow]{state['owner_name']}[/yellow]"))
    
    # Use webpage_scores rather than search_results if they exist
    if state.get("webpage_scores") and len(state["webpage_scores"]) > 0:
        relevant_results = state["webpage_scores"]
        logger.info(f"Using {len(relevant_results)} pre-scored webpages for content extraction")
    elif not state["search_results"]:
        error_msg = f"No address search results for {state['owner_name']}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    else:
        # Define priority sites for address lookup
        priority_sites = [
            "whitepages.com", "spokeo.com", "intelius.com", 
            "beenverified.com", "truepeoplesearch.com", "fastpeoplesearch.com",
            "peoplefinders.com", "addresses.com", "usphonebook.com",
            "radaris.com", "publicrecords360.com", "searchpeoplefree.com",
            "property", "assessor", "recorder", "parcel", "voter",
            "linkedin.com", "facebook.com", "twitter.com", "instagram.com",
            "zillow.com", "redfin.com", "realtor.com", "homes.com",
            "home address", "residential address", "property records"
        ]
        
        # Sort results to prioritize people finder sites and property records
        def priority_score(result):
            url = result.get("link", "").lower()
            # Check if any priority terms are in the URL
            score = 0
            for i, term in enumerate(priority_sites):
                if term in url:
                    score += (len(priority_sites) - i)  # Higher score for earlier items in priority_sites
            return score
        
        # Sort search results by priority score (highest first)
        sorted_results = sorted(state["search_results"], key=priority_score, reverse=True)
        
        # Take up to MAX_SEARCH_RESULTS search results for better coverage with priority to people finder sites
        relevant_results = sorted_results[:MAX_SEARCH_RESULTS]
    
    logger.info(f"Extracting content from {len(relevant_results)} address search results in parallel (max {MAX_CONCURRENT_EXTRACTIONS} concurrent)")
    start_time = time.time()
    
    # Extract content in parallel with concurrency limit
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTIONS)
    
    async def wrapped_extract(i, result):
        #async with semaphore:
        return await extract_single_address_webpage(i, len(relevant_results), result)
    
    # Run extractions in parallel with controlled concurrency
    extract_tasks = [wrapped_extract(i, result) for i, result in enumerate(relevant_results, 1)]
    
    # Gather results and handle failures
    results = []
    for task in asyncio.as_completed(extract_tasks):
        try:
            result = await task
            results.append(result)
        except Exception as e:
            logger.error(f"Extract task failed: {str(e)}")
            # We continue processing other tasks even if one fails
    
    # If no content was extracted, this is a critical error
    if not results:
        error_msg = f"Failed to extract content from any webpages for {state['owner_name']}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Log results
    cached_count = sum(1 for result in results if result.get("cached", False))
    total_duration = time.time() - start_time
    logger.info(f"Address content extraction completed in {total_duration:.2f}s for {len(results)}/{len(relevant_results)} webpages ({cached_count} from cache)")
    
    # Display results summary
    success_rate = len(results) / len(relevant_results) if relevant_results else 0
    console.print(f"[green]✓[/green] Successfully extracted content from [bold]{len(results)}[/bold] address webpages ({success_rate:.0%} success rate) in [bold]{total_duration:.2f}s[/bold] ({cached_count} from cache)")
    
    # Update state with webpage contents
    return {
        **state,
        "webpage_contents": results
    }

# Function to extract address from a single webpage content
async def extract_address_from_webpage(i: int, total: int, webpage: Dict[str, Any], chain: Any, owner_name: str, business_name: str, business_state: str, business_zip: str) -> Optional[Dict[str, Any]]:
    """Extract address information from a single webpage."""
    start_time = time.time()
    url = webpage.get('url', '')
    if not url:
        raise ValueError("No URL found in webpage data")
    
    # Format URL for display
    display_url = url[:50] + "..." if len(url) > 50 else url
    
    logger.info(f"Analyzing address webpage {i}/{total}: {url}")
    
    # Check cache first
    cached_extraction = cache.get_address_extraction(
        owner_name, 
        business_name, 
        business_state, 
        business_zip,
        url
    )
    
    if cached_extraction:
        duration = time.time() - start_time
        logger.debug(f"Using cached address extraction for {owner_name} from {url}")
        
        if cached_extraction.get("address") and cached_extraction.get("address") != "Unknown":
            log_substep(f"Using cached address from {display_url}")
            log_substep(f"[green]→[/green] Address: {cached_extraction['address']}, {cached_extraction.get('city', '') or ''} {cached_extraction['state']} {cached_extraction['zip_code']} (Confidence: {cached_extraction['confidence_score']}/10)")
            return cached_extraction
        else:
            log_substep(f"No address in cache for {display_url}")
            return None
    
    # Check if the URL is from a people-finder site - information for the LLM
    people_finder_sites = [
        "whitepages.com", "intelius.com", "beenverified.com", 
        "spokeo.com", "truthfinder.com", "peoplesearch.com",
        "fastpeoplesearch.com", "peoplefinders.com", "411.com",
        "radaris.com", "publicrecords.com", "property-records.com"
    ]
    
    is_people_finder = any(site in url.lower() for site in people_finder_sites)
    
    # Use content from the webpage
    content = webpage.get("content", "")
    if not content:
        logger.warning(f"Empty content for {url}")
        raise ValueError(f"No content available for {url}")
    
    # Truncate content to prevent token limits (if needed)
    max_content_length = 20000
    if len(content) > max_content_length:
        content = content[:max_content_length]
        logger.debug(f"Truncated content from {len(webpage['content'])} to {max_content_length} characters")
    
    # Extract address information using the LCEL chain
    response = await chain.ainvoke({
        "person_name": owner_name,
        "business_name": business_name,
        "business_state": business_state,
        "business_zip": business_zip,
        "webpage_content": content,
        "source_url": url,
        "is_people_finder": is_people_finder
    })
    
    duration = time.time() - start_time
    
    # Convert Pydantic model to dict for JSON serialization
    if hasattr(response, "model_dump"):
        # For Pydantic v2
        result = response.model_dump()
    elif hasattr(response, "dict"):
        # For Pydantic v1
        result = response.dict()
    else:
        # Manual conversion
        result = {
            "name": response.name if hasattr(response, "name") else owner_name,
            "address": response.address if hasattr(response, "address") else "Unknown",
            "city": response.city if hasattr(response, "city") else None,
            "state": response.state if hasattr(response, "state") else business_state,
            "zip_code": response.zip_code if hasattr(response, "zip_code") else business_zip,
            "confidence_score": response.confidence_score if hasattr(response, "confidence_score") else 0,
            "source_url": url,
            "rationale": response.rationale if hasattr(response, "rationale") else ""
        }
    
    # Cache the extraction result
    cache.cache_address_extraction(
        owner_name,
        business_name,
        business_state,
        business_zip,
        url,
        result
    )
    
    # Log extraction results
    if result.get("address") and result["address"] != "Unknown":
        logger.info(f"Found address for {owner_name} on {url}")
        log_substep(f"Found address in {duration:.2f}s from {display_url}")
        
        # Add source info to help with validation
        source_note = " (people finder site)" if is_people_finder else ""
        log_substep(f"[green]→[/green] Address: {result['address']}, {result.get('city', '') or ''} {result['state']} {result['zip_code']} (Confidence: {result['confidence_score']}/10){source_note}")
        
        return result
    else:
        logger.debug(f"No address found for {owner_name} on {url} in {duration:.2f}s")
        log_substep(f"No address found on {display_url}")
        return None

# Node 4: Extract addresses from webpage content
async def extract_addresses(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract addresses from extracted webpage content.
    """
    # Import needed utilities
    from core.utils.webpage import is_people_search_site
    
    start_time = time.time()
    business_name = state.get("business_name", "")
    business_state = state.get("business_state", "")
    business_zip = state.get("business_zip", "")
    contents = state.get("webpage_contents", [])
    
    console.print(Panel(f"[bold blue]Extracting Address Information for[/bold blue] [yellow]{business_name}[/yellow]"))
    
    # Skip if no content
    if not contents:
        logger.warning(f"No content to extract addresses from for {business_name}")
        return {
            **state,
            "owner_addresses": []
        }
    
    # Deduplicate content by URL to avoid processing the same page multiple times
    seen_urls = set()
    unique_contents = []
    
    for content in contents:
        # Extract URL safely, handle missing URL
        url = content.get("url", "") if isinstance(content, dict) else ""
        
        # Skip if no URL or already seen
        if not url:
            continue
            
        # Normalize URL to avoid minor variations
        normalized_url = url.rstrip("/").lower()
        
        # Skip duplicates
        if normalized_url in seen_urls:
            logger.info(f"Skipping duplicate URL for address extraction: {url}")
            continue
            
        # Add to unique contents and track URL
        seen_urls.add(normalized_url)
        unique_contents.append(content)
    
    # Log deduplication results
    duplicate_count = len(contents) - len(unique_contents)
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate URLs from {len(contents)} total content items")
    
    # Update contents with deduplicated list
    contents = unique_contents
    
    # Track progress for display
    progress = {"processed": 0, "total": len(contents), "addresses_found": 0}
    
    async def update_progress(i, total):
        progress["processed"] = i
        console.print(f"[green]Progress:[/green] Processed {i}/{total} content items, found {progress['addresses_found']} addresses", end="\r")
    
    # Process all content in parallel
    all_extracted_addresses = []
    
    # Add index and total count to each content item for logging
    for i, content_item in enumerate(contents):
        if isinstance(content_item, dict):
            content_item["index"] = i
            content_item["total"] = len(contents)
    
    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(10)  # Process up to 10 content items at once
    
    async def process_with_semaphore(content_item):
        async with semaphore:
            try:
                # For people search sites, use a special extractor to handle 403/429 errors
                if isinstance(content_item, dict) and content_item.get("url") and is_people_search_site(content_item.get("url", "")):
                    return await extract_addresses_from_people_search(
                        content_item=content_item,
                        business_name=business_name,
                        business_state=business_state,
                        business_zip=business_zip
                    )
                
                # For normal content, use the standard extractor
                addresses = await extract_single_address_content(
                    content_item,
                    business_name,
                    business_state,
                    business_zip,
                    progress_callback=update_progress
                )
                
                # Update progress
                if addresses:
                    progress["addresses_found"] += len(addresses)
                
                return addresses
            except Exception as e:
                logger.error(f"Error extracting addresses from content: {str(e)}")
                return []
    
    # Create tasks
    tasks = [process_with_semaphore(content) for content in contents]
    
    # Execute all tasks and gather results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results, handling any exceptions
    for result in results:
        if isinstance(result, list):
            all_extracted_addresses.extend(result)
        elif not isinstance(result, Exception):
            logger.warning(f"Unexpected result type from address extraction: {type(result)}")
    
    # Deduplicate addresses by normalizing and comparing
    unique_addresses = {}
    
    for address in all_extracted_addresses:
        if not isinstance(address, dict):
            continue
            
        # Normalize the address fields
        street = address.get("street", "").strip().lower() if address.get("street") else ""
        city = address.get("city", "").strip().lower() if address.get("city") else ""
        state = address.get("state", "").strip().lower() if address.get("state") else ""
        zip_code = address.get("zip", "").strip() if address.get("zip") else ""
        
        # Skip if missing key fields
        if not street or not state:
            continue
            
        # Create a key for deduplication
        address_key = f"{street}|{city}|{state}|{zip_code}"
        
        # Keep the one with the highest confidence or most complete data
        if address_key not in unique_addresses or address.get("confidence", 0) > unique_addresses[address_key].get("confidence", 0):
            unique_addresses[address_key] = address
    
    # Convert back to list
    final_addresses = list(unique_addresses.values())
    
    # Sort by confidence score (higher first)
    final_addresses.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    # Log results
    duration = time.time() - start_time
    logger.info(f"Extracted {len(final_addresses)} unique addresses from {len(contents)} content items in {duration:.2f}s")
    console.print(f"\n[green]✓[/green] Extracted [bold]{len(final_addresses)}[/bold] unique addresses in [bold]{duration:.2f}s[/bold]")
    
    # If found addresses, print them
    if final_addresses:
        for i, addr in enumerate(final_addresses[:5], 1):  # Show top 5
            console.print(f"  [yellow]{i}.[/yellow] {addr.get('full_address', 'Unknown address')} [grey](Confidence: {addr.get('confidence', 0)})[/grey]")
        
        if len(final_addresses) > 5:
            console.print(f"  ... and {len(final_addresses) - 5} more addresses")
    
    return {
        **state,
        "owner_addresses": final_addresses
    }

# Node 5: Synthesize final address
async def synthesize_final_address(state: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize the final address from extracted addresses with enhanced reasoning."""
    console.print(Panel(f"[bold blue]Synthesizing Final Address for[/bold blue] [yellow]{state['owner_name']}[/yellow]"))
    
    extracted_addresses = state.get("extracted_addresses", [])
    
    if not extracted_addresses:
        error_msg = f"No extracted addresses to synthesize for {state['owner_name']}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Gather all context for reasoning
    business_context = {
        "owner_name": state["owner_name"],
        "business_name": state["business_name"],
        "business_state": state["business_state"], 
        "business_zip": state["business_zip"]
    }
    
    start_time = time.time()
    
    logger.info(f"Synthesizing final address from {len(extracted_addresses)} extracted addresses")
    
    # Step 1: Group similar addresses using string similarity
    grouped_addresses = group_similar_addresses(extracted_addresses)
    
    # Step 2: Score each address group with enhanced contextual reasoning
    scored_groups = []
    
    for group in grouped_addresses:
        score = score_address_group(group, business_context)
        scored_groups.append((group, score))
    
    # Sort by score (highest first)
    scored_groups.sort(key=lambda x: x[1]["total_score"], reverse=True)
    
    # Log the results of scoring
    logger.info(f"Scored {len(scored_groups)} address groups")
    for i, (group, score) in enumerate(scored_groups):
        sample_address = group[0]
        logger.info(f"  Group {i+1}: Score {score['total_score']:.2f} - {sample_address.get('address', 'Unknown')} (Confidence: {score['confidence_score']}/10)")
        logger.info(f"    Evidence: {score['evidence_summary']}")
    
    # Get the highest-scoring address group
    if not scored_groups:
        error_msg = f"No valid address groups for {state['owner_name']}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    best_group, best_score = scored_groups[0]
    
    # Step 3: Construct the final address from the best group
    final_address = construct_final_address(best_group, best_score, business_context)
    
    # Step 4: Validate the final address for consistency and completeness
    validation_issues = validate_address(final_address, business_context)
    if validation_issues:
        logger.warning(f"Validation issues with final address: {', '.join(validation_issues)}")
        # Add validation issues to the final address
        final_address["validation_issues"] = validation_issues
    
    # Calculate duration
    duration = time.time() - start_time
    logger.info(f"Address synthesis completed in {duration:.2f}s")
    
    # Log the result summary
    full_address = f"{final_address.get('address', 'Unknown')}, {final_address.get('city', '') or ''} {final_address.get('state', '')} {final_address.get('zip_code', '')}"
    console.print(f"[green]✓[/green] Final address: [bold]{full_address}[/bold] (Confidence: {final_address['confidence_score']}/10)")
    
    # Update state with final address
    return {
        **state,
        "final_address": final_address
    }

def group_similar_addresses(addresses: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group similar addresses together using string similarity and context.
    
    Args:
        addresses: List of extracted address dictionaries
        
    Returns:
        List of address groups (each group is a list of similar addresses)
    """
    if not addresses:
        return []
    
    import re
    from difflib import SequenceMatcher
    
    # Helper function to normalize address for comparison
    def normalize_address(addr: str) -> str:
        """Normalize address for comparison."""
        # Convert to lowercase
        addr = addr.lower()
        
        # Replace apartment/unit designators with standard format
        addr = re.sub(r'(apt|apartment|unit|suite|ste|#)\s*', 'apt ', addr)
        
        # Remove punctuation except for apt numbers
        addr = re.sub(r'[^\w\s#-]', '', addr)
        
        # Standardize common abbreviations
        replacements = {
            'street': 'st',
            'avenue': 'ave',
            'road': 'rd',
            'boulevard': 'blvd',
            'drive': 'dr',
            'lane': 'ln',
            'place': 'pl',
            'court': 'ct',
            'circle': 'cir',
            'highway': 'hwy',
        }
        
        for word, abbr in replacements.items():
            addr = re.sub(r'\b' + word + r'\b', abbr, addr)
            
        # Standardize directional abbreviations
        directions = {
            'north': 'n',
            'south': 's',
            'east': 'e',
            'west': 'w',
            'northeast': 'ne',
            'northwest': 'nw',
            'southeast': 'se',
            'southwest': 'sw',
        }
        
        for word, abbr in directions.items():
            addr = re.sub(r'\b' + word + r'\b', abbr, addr)
            
        return ' '.join(addr.split())
    
    # Calculate address similarity score (0-1)
    def address_similarity(addr1: str, addr2: str) -> float:
        """Calculate similarity between two addresses."""
        if not addr1 or not addr2:
            return 0.0
            
        # Normalize both addresses
        norm1 = normalize_address(addr1)
        norm2 = normalize_address(addr2)
        
        # Get the first part of the address (usually house number and street)
        # This is more important for matching than city/state/zip
        parts1 = norm1.split(',')[0].strip() if ',' in norm1 else norm1
        parts2 = norm2.split(',')[0].strip() if ',' in norm2 else norm2
        
        # Calculate similarity score
        return SequenceMatcher(None, parts1, parts2).ratio()
    
    # Group similar addresses
    groups = []
    
    for address in addresses:
        addr_str = address.get('address', '')
        if not addr_str:
            continue
            
        # Try to find a matching group
        matched = False
        
        for group in groups:
            # Compare with the first address in the group
            group_addr = group[0].get('address', '')
            
            # Check similarity threshold (0.8 is fairly strict but allows for minor variations)
            if address_similarity(addr_str, group_addr) >= 0.8:
                group.append(address)
                matched = True
                break
        
        # Create a new group if no match found
        if not matched:
            groups.append([address])
    
    # Log grouping results
    logger.info(f"Grouped {len(addresses)} addresses into {len(groups)} groups")
    
    return groups

def score_address_group(
    address_group: List[Dict[str, Any]], 
    business_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Score an address group based on multiple factors with enhanced reasoning.
    
    Args:
        address_group: List of similar addresses in a group
        business_context: Business and owner context information
        
    Returns:
        Dictionary with scores and reasoning
    """
    if not address_group:
        return {
            "total_score": 0.0,
            "confidence_score": 0,
            "evidence_summary": "Empty address group"
        }
    
    # Extract context
    owner_name = business_context["owner_name"]
    business_state = business_context["business_state"]
    
    # Initialize scoring factors
    base_confidence = 0
    recency_score = 0
    locality_score = 0
    source_quality_score = 0
    evidence_strength_score = 0
    mention_frequency_score = 0
    validation_score = 0
    
    evidence_points = []
    
    # 1. Consider the highest original confidence score as base
    base_confidence = max([a.get('confidence_score', 0) for a in address_group])
    evidence_points.append(f"Highest original confidence: {base_confidence}")
    
    # 2. Recency: Check if addresses appear recent
    # Currently we don't have direct date info, but could be added in future
    recency_score = 0.5  # Neutral score without date information
    
    # 3. Locality: Check if address is in the same state as business
    address_states = [a.get('state', '') for a in address_group if a.get('state')]
    if address_states:
        # Count how many addresses match the business state
        matching_states = sum(1 for state in address_states if state == business_state)
        if matching_states > 0:
            # 1.0 if all match, lower if some don't match
            locality_score = matching_states / len(address_states)
            if locality_score == 1.0:
                evidence_points.append(f"All addresses in business state ({business_state})")
            else:
                evidence_points.append(f"{matching_states}/{len(address_states)} addresses in business state")
        else:
            locality_score = 0.0
            evidence_points.append(f"No addresses in business state (found in {', '.join(set(address_states))})")
    
    # 4. Source quality: Evaluate sources (basic version - could be enhanced)
    source_urls = [a.get('source_url', '') for a in address_group if a.get('source_url')]
    unique_sources = len(set(source_urls))
    
    # More unique sources = higher quality
    if unique_sources > 2:
        source_quality_score = 1.0
        evidence_points.append(f"High-quality evidence from {unique_sources} different sources")
    elif unique_sources > 0:
        source_quality_score = 0.6
        evidence_points.append(f"Evidence from {unique_sources} sources")
    else:
        source_quality_score = 0.3
        evidence_points.append("No source URLs provided")
    
    # 5. Evidence strength: Check owner name matches in context
    owner_matches_lists = [a.get('owner_matches', []) for a in address_group if a.get('owner_matches')]
    if owner_matches_lists:
        # Flatten the list of lists
        all_owner_matches = [match for sublist in owner_matches_lists for match in sublist]
        
        # Calculate match strength based on exact matches of the owner name
        exact_matches = sum(1 for match in all_owner_matches if owner_name.lower() in match.lower())
        if exact_matches > 0:
            evidence_strength_score = min(1.0, exact_matches / 3)  # Cap at 1.0
            evidence_points.append(f"Found {exact_matches} direct references to {owner_name}")
        else:
            evidence_strength_score = 0.4  # Some evidence but not direct
            evidence_points.append(f"No direct references to {owner_name}")
    else:
        evidence_strength_score = 0.2
        evidence_points.append("No owner name matches provided")
    
    # 6. Mention frequency: Reward addresses that appear multiple times
    mention_frequency_score = min(1.0, len(address_group) / 3)  # Cap at 1.0
    evidence_points.append(f"Address appears {len(address_group)} times")
    
    # 7. Validation: Check if addresses have been validated
    is_validated = any(a.get('is_residential', False) for a in address_group)
    if is_validated:
        validation_score = 1.0
        evidence_points.append("Confirmed as residential address")
    else:
        validation_score = 0.5
        evidence_points.append("Not explicitly confirmed as residential")
    
    # Calculate weighted total score
    weights = {
        "base_confidence": 3.0,  # Most important factor
        "locality": 2.0,         # High importance for state match
        "evidence_strength": 2.0, # High importance for owner connection
        "validation": 1.5,       # Important for residential confirmation
        "mention_frequency": 1.0, # Moderate importance
        "source_quality": 1.0,    # Moderate importance
        "recency": 0.5            # Lower importance without direct data
    }
    
    weighted_scores = [
        (base_confidence / 10) * weights["base_confidence"],  # Normalize to 0-1
        locality_score * weights["locality"],
        evidence_strength_score * weights["evidence_strength"],
        validation_score * weights["validation"],
        mention_frequency_score * weights["mention_frequency"],
        source_quality_score * weights["source_quality"],
        recency_score * weights["recency"]
    ]
    
    # Calculate total score (0-10 scale)
    max_possible_score = sum(weights.values())
    total_score = sum(weighted_scores) / max_possible_score * 10
    
    # Calculate confidence score (0-10 scale, integer)
    confidence_score = min(10, max(1, round(total_score)))
    
    # Create evidence summary
    evidence_summary = "; ".join(evidence_points)
    
    return {
        "total_score": total_score,
        "confidence_score": confidence_score,
        "evidence_summary": evidence_summary,
        "detail_scores": {
            "base_confidence": base_confidence,
            "locality_score": locality_score,
            "evidence_strength_score": evidence_strength_score,
            "validation_score": validation_score,
            "mention_frequency_score": mention_frequency_score,
            "source_quality_score": source_quality_score,
            "recency_score": recency_score
        }
    }

def construct_final_address(
    address_group: List[Dict[str, Any]],
    score: Dict[str, Any],
    business_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Construct the final address from the best address group with enhanced completeness.
    
    Args:
        address_group: List of similar addresses in the best group
        score: Score information for the address group
        business_context: Business and owner context
        
    Returns:
        Final address dictionary
    """
    if not address_group:
        return {
            "address": "Unknown",
            "city": None,
            "state": business_context["business_state"],
            "zip_code": business_context["business_zip"],
            "confidence_score": 1,
            "rationale": "No valid address found",
            "source_url": ""
        }
    
    # Select the most complete address from the group
    best_address = None
    max_completeness = -1
    
    for addr in address_group:
        # Calculate completeness score based on presence of fields
        completeness = 0
        if addr.get('address'):
            completeness += 3  # Street address is most important
        if addr.get('city'):
            completeness += 2
        if addr.get('state'):
            completeness += 2
        if addr.get('zip_code'):
            completeness += 2
        
        # Additional info is good too
        if addr.get('source_url'):
            completeness += 1
        if addr.get('rationale'):
            completeness += 1
        if addr.get('relevance_reasoning'):
            completeness += 1
        
        # Update best if this is more complete
        if completeness > max_completeness:
            max_completeness = completeness
            best_address = addr
    
    # Now combine information from all addresses in the group to fill gaps
    result = {
        "address": best_address.get('address', 'Unknown'),
        "confidence_score": score["confidence_score"],
        "rationale": score["evidence_summary"],
        "source_url": best_address.get('source_url', '')
    }
    
    # Fill in city if available
    cities = [a.get('city') for a in address_group if a.get('city')]
    if cities:
        result["city"] = cities[0]  # Use the first available city
    else:
        result["city"] = None
    
    # Fill in state (use business state as fallback)
    states = [a.get('state') for a in address_group if a.get('state')]
    if states:
        result["state"] = states[0]  # Use the first available state
    else:
        result["state"] = business_context["business_state"]
    
    # Fill in zip code (use business zip as fallback)
    zip_codes = [a.get('zip_code') for a in address_group if a.get('zip_code')]
    if zip_codes:
        result["zip_code"] = zip_codes[0]  # Use the first available zip
    else:
        result["zip_code"] = business_context["business_zip"]
    
    # Add source information
    source_urls = [a.get('source_url') for a in address_group if a.get('source_url')]
    if len(source_urls) > 1:
        result["additional_sources"] = source_urls[1:]
    
    return result

def validate_address(address: Dict[str, Any], business_context: Dict[str, Any]) -> List[str]:
    """Validate the final address for consistency and completeness.
    
    Args:
        address: Final address dictionary
        business_context: Business and owner context
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    # Check for missing essential fields
    if not address.get('address') or address.get('address') == 'Unknown':
        issues.append("Missing street address")
    
    # Check state consistency with business
    if address.get('state') != business_context["business_state"]:
        issues.append(f"Address state ({address.get('state')}) does not match business state ({business_context['business_state']})")
    
    # Verify zip code format
    zip_code = address.get('zip_code', '')
    if not zip_code:
        issues.append("Missing zip code")
    elif not (zip_code.isdigit() and (len(zip_code) == 5 or len(zip_code) == 9)):
        issues.append(f"Invalid zip code format: {zip_code}")
    
    # Low confidence warning
    if address.get('confidence_score', 0) < 5:
        issues.append(f"Low confidence score: {address.get('confidence_score', 0)}/10")
    
    return issues

# Create the address search graph
def create_address_search_graph() -> StateGraph:
    """Create the LangGraph workflow for address search."""
    # Initialize graph
    workflow = StateGraph(AddressSearchState)
    
    # Add nodes
    workflow.add_node("generate_address_queries", generate_address_queries)
    workflow.add_node("execute_address_searches", execute_address_searches)
    workflow.add_node("score_address_webpages", score_address_webpages)
    workflow.add_node("extract_address_webpage_content", extract_address_webpage_content)
    workflow.add_node("extract_addresses", extract_addresses)
    workflow.add_node("synthesize_final_address", synthesize_final_address)
    
    # Define edges
    workflow.add_edge("generate_address_queries", "execute_address_searches")
    workflow.add_edge("execute_address_searches", "score_address_webpages")
    workflow.add_edge("score_address_webpages", "extract_address_webpage_content")
    workflow.add_edge("extract_address_webpage_content", "extract_addresses")
    workflow.add_edge("extract_addresses", "synthesize_final_address")
    workflow.add_edge("synthesize_final_address", END)
    
    # Set entry point
    workflow.set_entry_point("generate_address_queries")
    
    return workflow 

def get_address_extraction_prompt() -> ChatPromptTemplate:
    """Get prompt for extracting addresses from webpage content with enhanced reasoning."""
    template = """You are an expert in extracting residential addresses from web content with perfect accuracy and detailed reasoning.

## TASK
Find any potential RESIDENTIAL addresses for {owner_name} in the text. {owner_name} owns a business called {business_name} in {business_state} with zip code {business_zip}.

## FOCUS
Focus exclusively on finding {owner_name}'s HOME address (residential address), NOT the business address.

## SEARCH PATTERNS
Look carefully for these indicators of residential addresses:
1. Direct statements connecting {owner_name} to a residential location (e.g., "lives at", "resides at", "home of", etc.)
2. Property records, tax records, or voter registration information 
3. Biographical information mentioning where {owner_name} lives
4. Real estate transactions involving {owner_name}
5. Public records directories that list {owner_name}'s personal address
6. Contextual clues differentiating between business and personal addresses

## ADDRESS VALIDATION CRITERIA
For each potential address, assess:
1. OWNERSHIP EVIDENCE: How strongly is the address linked to {owner_name} specifically?
2. RESIDENCE CONFIRMATION: Is this clearly a residential address (not business)?
3. RECENCY: Does the information appear current/recent?
4. CORROBORATION: Is the same address mentioned multiple times or across sources?
5. GEOGRAPHICAL CONSISTENCY: Is the address in or near {business_state}?

## TEXT TO ANALYZE:
{text}

## OUTPUT INSTRUCTIONS
{format_instructions}

## REASONING METHODOLOGY
1. First, identify ALL possible addresses in the text
2. For each address, evaluate its connection to {owner_name}
3. Assess each address against all validation criteria
4. Document your reasoning for including/excluding each address
5. Provide confidence levels based on evidence quality

If no valid residential addresses are found, return an empty list for addresses.
"""
    return ChatPromptTemplate.from_template(template) 

async def process_content(content: Dict[str, Any], business_name: str, business_state: str, business_zip: str, progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    Extract addresses from a content item.
    
    Args:
        content: Content dictionary containing text, source URL, etc.
        business_name: Name of the business
        business_state: State of the business
        business_zip: ZIP code of the business
        progress_callback: Optional callback function to report progress
        
    Returns:
        List of dictionaries with address information
    """
    content_text = content.get("content", "")
    content_url = content.get("url", "unknown")
    
    # Skip content that's too small to be useful
    if not content_text or len(content_text.strip()) < 50:
        logger.debug(f"Content from {content_url} is too small ({len(content_text) if content_text else 0} chars), skipping")
        return []
    
    # Log what we're processing
    i = int(content.get("index", 0)) + 1  # Convert to 1-indexed for display
    total = int(content.get("total", 1))
    logger.info(f"Extracting addresses from content {i}/{total}: {content_url}")
    
    addresses = await extract_addresses_from_text(
        content_text=content_text,
        content_url=content_url,
        business_name=business_name,
        business_state=business_state,
        business_zip=business_zip
    )
    
    # Report progress if callback is provided
    if progress_callback:
        try:
            await progress_callback(i, total)
        except Exception as e:
            logger.error(f"Error calling progress callback: {str(e)}")
    
    return addresses 