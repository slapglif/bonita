"""Workflow for business owner search using LangGraph compiler approach."""

import os
import traceback
import sys
import time
import asyncio
import re

from typing import Dict, List, Any, Optional, TypedDict, Annotated, Callable, cast
from pydantic import BaseModel, Field
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.utilities import SerpAPIWrapper
from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from core.utils.webpage import js_aware_loader, clean_html_content, httpx_loader, playwright_content_loader
from core.utils.crawler import recursive_crawl
from core.utils.logging import logger, log_substep
from core.utils.cache import SQLiteCache
from core.configs.llm_configs import build_llm

from core.prompts.lead_enrichment.business_owner_search import (
    BusinessOwner,
    BusinessOwnerList,
    SearchQuery,
    WebpageScore,
    WebpageContent,
    get_search_query_generation_prompt,
    get_webpage_relevance_scoring_prompt,
    get_owner_extraction_prompt,
    get_final_result_synthesis_prompt,
)

# Rich console for better formatting
console = Console()
cache = SQLiteCache()
# Maximum number of concurrent operations
MAX_CONCURRENT_SEARCHES = 50
MAX_CONCURRENT_SCORINGS = 100
MAX_CONCURRENT_EXTRACTIONS = 10

# More aggressive search and extraction settings
MAX_SEARCH_RESULTS = 30  # Reduced from 100 to limit resource usage
MAX_RETRIES = 3
MIN_RELEVANCE_THRESHOLD = 3  # Minimum relevance score (out of 10) for a webpage to be considered
MIN_CONFIDENCE_THRESHOLD = 2

# Initialize cache
cache = SQLiteCache()

# Define the state for our graph
class BusinessOwnerSearchState(TypedDict):
    """State for business owner search workflow."""
    business_name: str
    business_state: str
    business_zip: str
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    webpage_scores: List[Dict[str, Any]]
    webpage_contents: List[Dict[str, Any]]
    extracted_owners: List[Dict[str, Any]]
    final_owners: List[Dict[str, Any]]
    owner_addresses: List[Dict[str, Any]] 
    errors: List[str]


# Node 1: Generate search queries
async def generate_search_queries(state: BusinessOwnerSearchState) -> BusinessOwnerSearchState:
    """Generate search queries for finding business owner information."""
    console.print(Panel(f"[bold blue]Generating Search Queries for[/bold blue] [yellow]{state['business_name']}[/yellow]"))
    
    start_time = time.time()
    
    # Check cache first
    cache_key = f"search_queries_{state['business_name']}_{state['business_state']}_{state['business_zip']}"
    cached_queries = cache.get_search_results(cache_key)
    
    if cached_queries:
        logger.info(f"Using {len(cached_queries)} cached search queries for {state['business_name']}")
        duration = time.time() - start_time
        
        for i, query in enumerate(cached_queries, 1):
            log_substep(f"Query {i}: {query}")
            
        logger.debug(f"Retrieved cached search queries in {duration:.2f} seconds")
        
        return {
            **state,
            "search_queries": cached_queries
        }
    
    # Generate new queries if not in cache
    llm = build_llm()
    
    # Initialize the Pydantic output parser
    parser = PydanticOutputParser(pydantic_object=SearchQuery)
    format_instructions = parser.get_format_instructions()
    
    # Get the prompt
    prompt = get_search_query_generation_prompt()
    
    logger.info(f"Generating search queries for {state['business_name']}")
    
    # Run the prompt
    chain = prompt.partial(format_instructions=format_instructions) | llm | parser
    
    response = await chain.ainvoke({
        "business_name": state["business_name"],
        "business_state": state["business_state"],
        "business_zip": state["business_zip"]
    })
    
    # Log generated queries
    logger.info(f"Generated {len(response.queries)} search queries for {state['business_name']}")
    for i, query in enumerate(response.queries, 1):
        log_substep(f"Query {i}: {query}")
    
    duration = time.time() - start_time
    logger.debug(f"Search query generation took {duration:.2f} seconds")
    
    # Cache the generated queries
    cache.cache_search_results(cache_key, response.queries)
    
    # Update state with generated queries
    return {
        **state,
        "search_queries": response.queries
    }

# Function to execute a single search query
async def execute_single_query(query: str, search: SerpAPIWrapper) -> Dict[str, Any]:
    """Execute a single search query and return the results."""
    start_time = time.time()
    full_query = f"{query}"
    
    # Default empty result
    empty_result = {
        "query": full_query,
        "results": [],
        "count": 0,
        "time": time.time() - start_time,
        "cached": False
    }
    
    # First try to get from cache
    cached_results = None
    try:
        cached_data = cache.get_search_results(full_query)
        if isinstance(cached_data, list):
            cached_results = cached_data
    except Exception as e:
        logger.warning(f"Cache error: {str(e)}")
    
    # If we have valid cached results, use them
    if cached_results:
        query_time = time.time() - start_time
        results_count = len(cached_results)
        
        # Make sure all results have valid links
        for i in range(len(cached_results)):
            if isinstance(cached_results[i], dict):
                result = cached_results[i]
                if "link" not in result or not result["link"]:
                    # Find a URL from other fields
                    for field in ["url", "displayLink"]:
                        if field in result and result[field]:
                            result["link"] = result[field]
                            break
                    
                    # If still no link, use placeholder
                    if "link" not in result or not result["link"]:
                        result["link"] = "unknown-url"
        
        log_substep(f"Using {results_count} cached results for query: {full_query} ({query_time:.2f}s)")
        
        return {
            "query": full_query,
            "results": cached_results,
            "count": results_count,
            "time": query_time,
            "cached": True
        }
    
    # Nothing in cache, execute search
    try:
        # Execute search
        search_result = await search.aresults(full_query)
        
        # Calculate execution time
        query_time = time.time() - start_time
        
        # Check if we have organic results
        if not isinstance(search_result, dict) or "organic_results" not in search_result:
            log_substep(f"No organic results found for query: {full_query}")
            return empty_result
        
        # Extract and format results
        organic_results = search_result["organic_results"]
        if not isinstance(organic_results, list):
            log_substep(f"Invalid organic results format for query: {full_query}")
            return empty_result
        
        results = []
        for result in organic_results:
            if not isinstance(result, dict):
                continue
                
            # Ensure URL is properly set
            url = result.get("link", "")
            if not url:
                url = result.get("displayLink", "unknown-url")
            
            # Add formatted result
            results.append({
                "query": full_query,
                "title": result.get("title", ""),
                "link": url,
                "snippet": result.get("snippet", "")
            })
        
        # Log results
        results_count = len(results)
        log_substep(f"Found {results_count} results in {query_time:.2f}s")
        
        # Cache results if we have any
        if results:
            try:
                cache.cache_search_results(full_query, results)
            except Exception as e:
                logger.warning(f"Cache error: {str(e)}")
        
        # Return results
        return {
            "query": full_query,
            "results": results,
            "count": results_count,
            "time": query_time,
            "cached": False
        }
    except Exception as e:
        # Handle search errors
        error_msg = str(e)
        logger.error(f"Error executing search query '{query}': {error_msg}")
        
        # Return empty results with error
        empty_result["error"] = error_msg
        return empty_result

# Node 2: Execute search queries
async def execute_search_queries(state: BusinessOwnerSearchState) -> BusinessOwnerSearchState:
    """Execute search queries using SerpAPI."""
    console.print(Panel(f"[bold blue]Searching the Web for[/bold blue] [yellow]{state['business_name']}[/yellow]"))
    
    # Check if SERPAPI_API_KEY is set
    if not os.environ.get("SERPAPI_API_KEY"):
        logger.error("SERPAPI_API_KEY not set in environment")
        raise ValueError("SERPAPI_API_KEY environment variable is required but not set")
    
    # Initialize our custom SerpAPI wrapper
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
    
    logger.info(f"Executing {search_stats['total_queries']} search queries in parallel (max {MAX_CONCURRENT_SEARCHES} concurrent)")
    start_time = time.time()
    
    # Execute search queries in parallel using asyncio.gather with semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)
    
    async def wrapped_search(i, query):
        async with semaphore:
            logger.info(f"Executing search query ({i}/{search_stats['total_queries']}): {query}")
            # Add retry logic for search queries
            for retry in range(MAX_RETRIES):
                try:
                    result = await execute_single_query(query, search)
                    return result
                except Exception as e:
                    if retry < MAX_RETRIES - 1:
                        wait_time = 2 ** retry  # exponential backoff
                        logger.warning(f"Search error, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {MAX_RETRIES} retries: {str(e)}")
                        return {
                            "query": query,
                            "results": [],
                            "count": 0,
                            "time": 0,
                            "cached": False,
                            "error": str(e)
                        }
    
    # Run queries in parallel with controlled concurrency
    search_tasks = [wrapped_search(i, query) for i, query in enumerate(state["search_queries"], 1)]
    search_results = await asyncio.gather(*search_tasks)
    
    # Process results - check if result is a valid dict and if it doesn't have an error field
    for result in search_results:
        # Skip invalid results or those with errors
        if not result or not isinstance(result, dict):
            continue
            
        if "error" not in result:
            # Only process valid results without errors
            if "time" in result:
                search_stats["query_times"].append(result["time"])
            
            if "count" in result:
                search_stats["total_results"] += result["count"]
            
            if result.get("cached", False):
                search_stats["cached_count"] += 1
            
            if "results" in result and isinstance(result["results"], list):
                all_search_results.extend(result["results"])
    
    # Log search statistics
    total_duration = time.time() - start_time
    avg_time = sum(search_stats["query_times"]) / len(search_stats["query_times"]) if search_stats["query_times"] else 0
    logger.info(f"Search completed in {total_duration:.2f}s: {search_stats['total_results']} total results from {search_stats['total_queries']} queries ({search_stats['cached_count']} cached)")
    logger.debug(f"Average query time: {avg_time:.2f}s")
    
    # Display search summary
    console.print(f"[green]✓[/green] Found [bold]{search_stats['total_results']}[/bold] relevant results in [bold]{total_duration:.2f}s[/bold] ({search_stats['cached_count']} from cache)")
    
    # Check if we found enough results - if not but we have SOME, still proceed
    if search_stats['total_results'] == 0:
        error_msg = f"No search results found for {state['business_name']}. Need to expand search."
        logger.error(error_msg)
        return {
            **state,
            "search_results": [],
            "errors": state["errors"] + [error_msg]
        }
    
    # Update state with search results
    return {
        **state,
        "search_results": all_search_results
    }

def deduplicate_urls(url_list: List[Dict[str, Any]], url_field: str = "link") -> List[Dict[str, Any]]:
    """Deduplicate URLs in a list of dictionaries.
    
    Args:
        url_list: List of dictionaries containing URLs
        url_field: Field name containing the URL in each dictionary
        
    Returns:
        Deduplicated list of dictionaries
    """
    seen_urls = set()
    unique_results = []
    
    for item in url_list:
        # Skip items without the URL field
        if not isinstance(item, dict) or url_field not in item:
            continue
            
        url = item[url_field]
        
        # Skip empty URLs
        if not url:
            continue
            
        # Normalize URL to avoid minor variations (remove trailing slashes, etc)
        normalized_url = url.rstrip("/").lower()
        
        # Skip if we've seen this URL before
        if normalized_url in seen_urls:
            logger.debug(f"Skipping duplicate URL: {url}")
            continue
            
        # Add to seen URLs and unique results
        seen_urls.add(normalized_url)
        unique_results.append(item)
    
    duplicate_count = len(url_list) - len(unique_results)
    logger.info(f"Removed {duplicate_count} duplicate URLs from {len(url_list)} total results")
    
    return unique_results

# Function to score a single webpage
async def score_single_webpage(
    i: int, 
    total: int, 
    result: Dict[str, Any], 
    chain: Any, 
    business_name: str, 
    business_state: str, 
    business_zip: str
) -> Dict[str, Any]:
    """Score a single webpage for relevance."""
    item_start = time.time()
    
    # Get URL from result
    url = "unknown-url"  # Default fallback
    
    # Try to extract URL
    if isinstance(result, dict):
        for key in ["link", "url", "displayLink"]:
            if key in result and result[key]:
                url = str(result[key])
                break
    
    logger.debug(f"Processing URL for scoring: {url}")
    
    # Check cache first
    cache_key = f"webpage_score_{url}_{business_name}_{business_state}_{business_zip}"
    cached_score = cache.get_search_results(cache_key)
    
    if cached_score and isinstance(cached_score, dict):
        item_duration = time.time() - item_start
        owner_indicator = "✓" if cached_score.get("owner_likely_present", False) else "✗"
        
        # Make sure URL is explicitly stored
        cached_score["url"] = url
        
        # Format the URL for display - avoid using Text() which might show None
        display_url = url[:50] + "..." if len(url) > 50 else url
        log_substep(f"Using cached score for {display_url} - Relevance: {cached_score['relevance_score']}/10, Owner likely: {owner_indicator} ({item_duration:.2f}s)")
        return cached_score
    
    try:
        logger.debug(f"Scoring webpage {i}/{total}: {url}")
        
        # Get content for scoring
        snippet = ""
        if isinstance(result, dict):
            snippet = result.get("snippet", result.get("title", ""))
        
        # Score the webpage
        response = await chain.ainvoke({
            "business_name": business_name,
            "business_state": business_state,
            "business_zip": business_zip,
            "webpage_url": url,
            "webpage_content": snippet
        })
        
        item_duration = time.time() - item_start
        owner_indicator = "✓" if response.owner_likely_present else "✗"
        
        # Format the URL for display directly without using Text()
        display_url = url[:50] + "..." if len(url) > 50 else url
        log_substep(f"Scored {display_url} - Relevance: {response.relevance_score}/10, Owner likely: {owner_indicator} ({item_duration:.2f}s)")
        
        # Create result with explicit URL
        score_result = {
            "url": url,
            "title": result.get("title", ""),
            "relevance_score": response.relevance_score,
            "owner_likely_present": response.owner_likely_present,
            "reasoning": response.reasoning,
            "duration": item_duration
        }
        
        # Cache the result
        cache.cache_search_results(cache_key, score_result)
        
        return score_result
    except Exception as e:
        item_duration = time.time() - item_start
        logger.error(f"Error scoring webpage {url}: {str(e)}")
        
        # Fail loudly 
        raise ValueError(f"Error scoring webpage {url}: {str(e)}")

# Node 3: Score webpages for relevance
async def score_webpage_relevance(state: BusinessOwnerSearchState) -> BusinessOwnerSearchState:
    """Score webpages for relevance to business owner information."""
    console.print(Panel(f"[bold blue]Scoring Webpage Relevance for[/bold blue] [yellow]{state['business_name']}[/yellow]"))
    
    if not state["search_results"]:
        error_msg = f"No search results to score for {state['business_name']}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    start_time = time.time()
    llm = build_llm()
    
    # Initialize the Pydantic output parser
    parser = PydanticOutputParser(pydantic_object=WebpageScore)
    format_instructions = parser.get_format_instructions()
    
    # Get the prompt
    prompt = get_webpage_relevance_scoring_prompt()
    
    # Create a proper LCEL chain
    chain = prompt.partial(format_instructions=format_instructions) | llm | parser
    
    # Deduplicate URLs using the utility function
    unique_results = deduplicate_urls(state["search_results"])
    
    logger.info(f"Scoring {len(unique_results)} unique webpages for relevance")
    
    # Score webpages in parallel with concurrency limit
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCORINGS)
    
    async def wrapped_score(i, result):
        #async with semaphore:
        return await score_single_webpage(
            i,
            len(unique_results),
            result,
            chain, 
            state['business_name'],
            state['business_state'],
            state['business_zip']
        )
    
    # Run scoring in parallel with controlled concurrency
    score_tasks = [wrapped_score(i, result) for i, result in enumerate(unique_results, 1)]
    
    # Gather results and handle failures
    all_scores = []
    for task in asyncio.as_completed(score_tasks):
        try:
            score = await task
            all_scores.append(score)
        except Exception as e:
            logger.error(f"Scoring task failed: {str(e)}")
            # Continue with other tasks
    
    # Use a higher minimum threshold for relevance
    MIN_RELEVANCE_THRESHOLD = 5  # Increase from 3 to 5 to get better quality results
    
    # Filter out low-scoring webpages using MIN_RELEVANCE_THRESHOLD
    relevant_scores = [
        score for score in all_scores 
        if score.get("relevance_score", 0) >= MIN_RELEVANCE_THRESHOLD
    ]
    
    # Sort remaining scores by relevance_score (highest first)
    webpage_scores = sorted(
        relevant_scores,
        key=lambda x: x.get("relevance_score", 0),
        reverse=True
    )
    
    # Use a dynamic approach to limit results based on score distribution
    # Only take pages with scores of 7+ or the top 15, whichever is smaller
    high_quality_threshold = 7
    high_quality_results = [s for s in webpage_scores if s.get("relevance_score", 0) >= high_quality_threshold]
    
    # If we have enough high-quality results, use only those
    if len(high_quality_results) >= 5:
        top_webpages = high_quality_results[:MAX_SEARCH_RESULTS]
    else:
        # Otherwise use top 15 results but no more than MAX_SEARCH_RESULTS
        MAX_RESULTS_TO_USE = min(15, MAX_SEARCH_RESULTS)
        top_webpages = webpage_scores[:MAX_RESULTS_TO_USE]
    
    # Log results
    total_duration = time.time() - start_time
    excluded_count = len(all_scores) - len(relevant_scores)
    limited_count = len(webpage_scores) - len(top_webpages)
    
    logger.info(f"Webpage scoring completed in {total_duration:.2f}s: selected {len(top_webpages)} out of {len(all_scores)} scored webpages")
    logger.info(f"Excluded {excluded_count} webpages with relevance scores below {MIN_RELEVANCE_THRESHOLD}")
    logger.info(f"Limited {limited_count} additional webpages to focus on highest quality results")
    
    # Display results summary
    console.print(f"[green]✓[/green] Selected [bold]{len(top_webpages)}[/bold] most relevant webpages in [bold]{total_duration:.2f}s[/bold] (excluded {excluded_count + limited_count} lower-relevance pages)")
    
    # If no relevant webpages, return warning but continue with a few highest-scoring ones as fallback
    if not top_webpages:
        logger.warning(f"No webpages scored above the minimum threshold of {MIN_RELEVANCE_THRESHOLD}/10 for {state['business_name']}")
        
        # Fall back to using the top 3 highest scoring pages regardless of threshold (reduced from 5)
        fallback_pages = sorted(
            all_scores,
            key=lambda x: x.get("relevance_score", 0),
            reverse=True
        )[:3]  # Reduced fallback count from 5 to 3
        
        if fallback_pages:
            logger.info(f"Using {len(fallback_pages)} highest scoring pages as fallback")
            console.print(f"[yellow]⚠[/yellow] Using {len(fallback_pages)} highest scoring pages as fallback (scores below threshold)")
            top_webpages = fallback_pages
        else:
            error_msg = f"No scored webpages available for {state['business_name']}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Update state with webpage scores
    return {
        **state,
        "webpage_scores": top_webpages
    }

# Function to extract content from a single webpage
async def extract_single_webpage(i: int, total: int, webpage: Dict[str, Any]) -> Dict[str, Any]:
    """Extract content from a single webpage."""
    # Import Document class at the function level to ensure it's available
    from langchain_core.documents import Document
    
    # Start a timer to track overall execution time
    start_time = time.time()
    
    # Setup an absolute timeout for this function to prevent hanging
    absolute_timeout = 120  # 2 minutes max total time
    
    # Create a safety timeout task that will terminate this function if it runs too long
    safety_timeout_task = None
    
    try:
        # Ensure we have a valid URL
        url = None
        for field in ["url", "link", "displayLink"]:
            if field in webpage and webpage[field]:
                url = webpage[field]
                break
        
        # If no valid URL, this is an error - we cannot proceed
        if not url:
            raise ValueError("No URL found in webpage data")
        
        # Ensure url is a string
        url = str(url)
        
        # Create a safety timeout task that will be triggered if the function takes too long
        async def safety_timeout():
            await asyncio.sleep(absolute_timeout)
            # If we get here, the function has been running too long - log and terminate
            logger.error(f"Extraction timed out after {absolute_timeout}s for {url}")
            # Use os._exit for emergency termination if needed
            # os._exit(1)
            # Instead of hard termination, raise a timeout exception
            raise asyncio.TimeoutError(f"Extraction process exceeded {absolute_timeout}s timeout for {url}")
            
        # Start the safety timeout task
        safety_timeout_task = asyncio.create_task(safety_timeout())
        
        # Check for document file extensions we should handle differently
        document_extensions = ['.xlsx', '.xls', '.doc', '.docx', '.pdf', '.zip', '.rar', '.7z', '.csv']
        is_document_url = any(url.lower().endswith(ext) for ext in document_extensions)
        
        # Format URL for display
        display_url = url[:50] + "..." if len(url) > 50 else url
        
        # Get relevance score for context
        relevance_score = webpage.get("relevance_score", 0)
        
        # Get extraction settings from webpage or use defaults
        timeout = webpage.get("timeout", 60)
        max_retries = webpage.get("retries", 2)
        is_anti_bot_site = webpage.get("is_anti_bot", False)
        
        # For document files, use a shorter timeout to prevent hanging
        if is_document_url:
            logger.info(f"Document file detected: {url} - using shorter timeout")
            timeout = min(timeout, 30)  # Cap timeout for document files
        
        logger.info(f"Extracting content from webpage {i}/{total}: {url} (score: {relevance_score}/10, timeout: {timeout}s)")
        
        # Check cache first - fast path to avoid network call
        cached_content = cache.get_webpage_content(url)
        if cached_content:
            duration = time.time() - start_time
            content_length = len(cached_content["content"])
            logger.debug(f"Using cached content for {url}: {content_length} characters")
            
            log_substep(f"Using cached content ({content_length} chars) from {display_url}")
            
            # Cancel the safety timeout
            if safety_timeout_task:
                safety_timeout_task.cancel()
            
            return {
                "url": url,
                "title": cached_content.get("title", webpage.get("title", "")),
                "content": cached_content["content"],
                "success": True,
                "score": relevance_score,
                "duration": duration,
                "cached": True,
                "is_pdf": cached_content.get("is_pdf", False),
                "query": webpage.get("query", None)  # Preserve the query
            }
        
        # Set a task-specific timeout for this extraction
        extraction_timeout = min(timeout, absolute_timeout - 10)  # Give buffer before absolute timeout
        
        # Extract new content if not in cache
        try:
            # Handle document file types - quick return path for known document types
            if is_document_url:
                # Check if URL looks like a PDF
                is_pdf = url.lower().endswith('.pdf')
                
                # Handle PDF files with our PDF extractor
                if is_pdf:
                    logger.info(f"Detected PDF URL: {url}")
                    
                    # Use a timeout for PDF extraction to prevent hanging
                    pdf_timeout = min(timeout, 60)  # 60s max for PDFs
                    
                    try:
                        # Create a task for PDF extraction with timeout
                        pdf_task = asyncio.create_task(extract_pdf_document(url))
                        # Wait for the task with timeout
                        pdf_content = await asyncio.wait_for(pdf_task, timeout=pdf_timeout)
                        
                        if pdf_content and pdf_content.get("content"):
                            content_length = len(pdf_content["content"])
                            duration = time.time() - start_time
                            
                            log_substep(f"Extracted PDF ({content_length} chars) in {duration:.2f}s from {display_url}")
                            
                            # Add PDF flag and other info
                            pdf_content["url"] = url
                            pdf_content["is_pdf"] = True
                            pdf_content["score"] = relevance_score
                            pdf_content["success"] = True
                            pdf_content["duration"] = duration
                            pdf_content["cached"] = False
                            pdf_content["query"] = webpage.get("query", None)  # Preserve the query
                            
                            # Cache the extracted content
                            cache.cache_webpage_content(url, pdf_content["content"], pdf_content.get("title", "PDF Document"))
                            
                            # Cancel the safety timeout
                            if safety_timeout_task:
                                safety_timeout_task.cancel()
                            
                            return pdf_content
                    except asyncio.TimeoutError:
                        logger.warning(f"PDF extraction timed out after {pdf_timeout}s for {url}")
                        # Fall through to standard document handling
                    except Exception as pdf_err:
                        logger.warning(f"PDF extraction failed: {str(pdf_err)}")
                        # Fall through to standard document handling
                
                # Create a stub document result for any document type
                logger.info(f"Document URL detected: {url} - creating stub")
                
                # Create a stub document result
                doc_type = url.split('.')[-1].upper()
                title = webpage.get("title", f"{doc_type} Document: {url.split('/')[-1]}")
                
                # Create stub content that acknowledges the document
                stub_content = f"Document URL: {url}\nDocument Type: {doc_type}\nTitle: {title}\n\nThis is a {doc_type} document that could not be directly parsed for text content. The document may contain relevant business information but requires specialized parsing not available in the current pipeline."
                
                duration = time.time() - start_time
                
                result = {
                    "url": url,
                    "title": title,
                    "content": stub_content,
                    "success": True,
                    "score": relevance_score,
                    "duration": duration,
                    "cached": False,
                    "is_document": True,
                    "document_type": doc_type,
                    "loader": "document_handler",
                    "query": webpage.get("query", None)
                }
                
                # Cache the stub content
                cache.cache_webpage_content(url, stub_content, title)
                
                # Cancel the safety timeout
                if safety_timeout_task:
                    safety_timeout_task.cancel()
                    
                return result
            
            # For non-document URLs, first try with httpx (faster, no JavaScript)
            logger.debug(f"Loading webpage with httpx: {url}")
            httpx_start = time.time()
            httpx_timeout = int(timeout * 0.7)  # Shorter timeout for httpx
            
            # Use asyncio.wait_for to enforce timeout
            try:
                httpx_task = asyncio.create_task(httpx_loader(url, timeout=httpx_timeout, query=webpage.get("query", None)))
                documents = await asyncio.wait_for(httpx_task, timeout=httpx_timeout)
                httpx_duration = time.time() - httpx_start
            except asyncio.TimeoutError:
                # Ensure Document class is imported locally here too
                from langchain_core.documents import Document
                
                logger.warning(f"HTTPX extraction timed out after {httpx_timeout}s for {url}")
                # Create an empty document to trigger Playwright fallback
                documents = [Document(page_content="", metadata={"requires_javascript": True, "url": url})]
                httpx_duration = time.time() - httpx_start
            
            # Check if httpx identified this as a document file from content-type
            if documents and documents[0].metadata.get("content_type") and any(
                typ in documents[0].metadata.get("content_type", "") 
                for typ in ["application/vnd.openxmlformats", "application/vnd.ms-", "application/msword"]
            ):
                logger.info(f"Document content-type detected via httpx: {documents[0].metadata.get('content_type')}")
                
                # Create a stub document result based on content type
                content_type = documents[0].metadata.get("content_type", "document/unknown")
                doc_format = "Excel" if "excel" in content_type else "Word" if "word" in content_type else "Office Document"
                
                title = webpage.get("title", f"{doc_format} Document: {url.split('/')[-1]}")
                
                # Create stub content that acknowledges the document
                stub_content = f"Document URL: {url}\nDocument Type: {doc_format}\nContent-Type: {content_type}\nTitle: {title}\n\nThis is a {doc_format} document that could not be directly parsed for text content. The document may contain relevant business information but requires specialized parsing not available in the current pipeline."
                
                duration = time.time() - start_time
                
                result = {
                    "url": url,
                    "title": title,
                    "content": stub_content,
                    "success": True,
                    "score": relevance_score,
                    "duration": duration,
                    "cached": False,
                    "is_document": True,
                    "document_type": doc_format,
                    "content_type": content_type,
                    "loader": "httpx_document_handler",
                    "query": webpage.get("query", None)
                }
                
                # Cache the stub content
                cache.cache_webpage_content(url, stub_content, title)
                
                # Cancel the safety timeout
                if safety_timeout_task:
                    safety_timeout_task.cancel()
                    
                return result
            
            # Check if httpx loader returned content or if page requires JavaScript
            if (documents and documents[0].page_content.strip() and 
                not documents[0].metadata.get("requires_javascript", False)):
                # Successful extraction with httpx
                content = documents[0].page_content
                content_length = len(content)
                logger.debug(f"Extracted {content_length} characters with httpx from {url} in {httpx_duration:.2f}s")
                
                duration = time.time() - start_time
                log_substep(f"Extracted {content_length} chars with httpx in {duration:.2f}s from {display_url}")
                
                # Cache the extracted content
                title = documents[0].metadata.get("title", webpage.get("title", ""))
                cache.cache_webpage_content(url, content, title)
                
                # Cancel the safety timeout
                if safety_timeout_task:
                    safety_timeout_task.cancel()
                    
                return {
                    "url": url,
                    "title": title,
                    "content": content,
                    "success": True,
                    "score": relevance_score,
                    "duration": duration,
                    "cached": False,
                    "is_pdf": False,
                    "loader": "httpx",
                    "query": webpage.get("query", None)  # Preserve the query
                }
            
            # If the content type indicates this is a document, don't try with Playwright
            if documents and documents[0].metadata.get("content_type") and "application/vnd.openxmlformats-officedocument" in documents[0].metadata.get("content_type", ""):
                logger.warning(f"Document detected - skipping Playwright for: {url}")
                
                # Create a stub for the document
                content_type = documents[0].metadata.get("content_type", "document/unknown")
                doc_format = "Excel" if "spreadsheet" in content_type else "Word" if "wordprocessing" in content_type else "Office Document"
                
                title = webpage.get("title", f"{doc_format} Document: {url.split('/')[-1]}")
                stub_content = f"Document URL: {url}\nDocument Type: {doc_format}\nContent-Type: {content_type}\n\nThis file is a {doc_format} document that cannot be processed by the text extraction pipeline."
                
                duration = time.time() - start_time
                
                result = {
                    "url": url,
                    "title": title,
                    "content": stub_content,
                    "success": True,
                    "score": relevance_score * 0.7,  # Reduce score for document stubs
                    "duration": duration,
                    "cached": False,
                    "is_document": True,
                    "document_type": doc_format,
                    "content_type": content_type,
                    "loader": "document_handler",
                    "query": webpage.get("query", None)
                }
                
                # Cache the stub
                cache.cache_webpage_content(url, stub_content, title)
                
                # Cancel the safety timeout
                if safety_timeout_task:
                    safety_timeout_task.cancel()
                    
                return result
            
            # If httpx failed or page requires JavaScript, use our optimized Playwright implementation
            playwright_reason = "Page requires JavaScript" if documents and documents[0].metadata.get("requires_javascript", False) else "HTTPX extraction failed"
            logger.info(f"{playwright_reason}, using Playwright for: {url}")
            
            # Use a reduced timeout for sites that might freeze
            safe_timeout = min(timeout, 45)  # Limit to 45 seconds max
            
            # Set optimized parameters using the values from the webpage object
            logger.debug(f"Using playwright_content_loader for {url} with timeout={safe_timeout}s, retries={max_retries}")
            
            # Use asyncio.wait_for to enforce timeout
            try:
                # Execute Playwright with timeout protection
                playwright_start = time.time()
                playwright_task = asyncio.create_task(playwright_content_loader(
                    url=url,
                    timeout=safe_timeout, 
                    max_retries=max_retries,
                    query=webpage.get("query", None)  # Pass the original query for semantic context
                ))
                documents = await asyncio.wait_for(playwright_task, timeout=safe_timeout + 15)  # Add small buffer
                playwright_duration = time.time() - playwright_start
            except asyncio.TimeoutError:
                logger.warning(f"Playwright extraction timed out after {safe_timeout}s for {url}")
                # Handle timeout as a failure and return stub document
                documents = []
                playwright_duration = time.time() - playwright_start
            
            # Check if Playwright extraction succeeded
            if documents and len(documents) > 0 and documents[0].page_content.strip():
                content = documents[0].page_content
                content_length = len(content)
                logger.debug(f"Extracted {content_length} characters with Playwright from {url} in {playwright_duration:.2f}s")
                
                duration = time.time() - start_time
                log_substep(f"Extracted {content_length} chars with Playwright in {duration:.2f}s from {display_url}")
                
                # Cache the extracted content
                title = documents[0].metadata.get("title", webpage.get("title", ""))
                cache.cache_webpage_content(url, content, title)
                
                # Cancel the safety timeout
                if safety_timeout_task:
                    safety_timeout_task.cancel()
                
                return {
                    "url": url,
                    "title": title,
                    "content": content,
                    "success": True,
                    "score": relevance_score,
                    "duration": duration,
                    "cached": False,
                    "is_pdf": False,
                    "loader": "playwright",
                    "is_anti_bot_site": is_anti_bot_site,
                    "query": webpage.get("query", None)  # Preserve the query
                }
            
            # Skip recursive crawling - it's often problematic for this use case
            logger.warning(f"Both httpx and Playwright extraction failed for {url}")
            
            # If all methods failed, create a stub
            duration = time.time() - start_time
            stub_content = f"Unable to extract content from {url} after attempting multiple methods."
            title = webpage.get("title", "Extraction Failed")
            
            # Create a stub for failed extraction
            result = {
                "url": url,
                "title": title,
                "content": stub_content,
                "success": True,  # Mark as success to avoid hard errors, but with empty content
                "score": relevance_score * 0.2,  # Significantly reduce score
                "duration": duration,
                "cached": False,
                "is_extraction_failure": True,
                "loader": "extraction_failure",
                "query": webpage.get("query", None)
            }
            
            # Cache the stub
            cache.cache_webpage_content(url, stub_content, title)
            
            # Cancel the safety timeout
            if safety_timeout_task:
                safety_timeout_task.cancel()
                
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error extracting content from {url}: {str(e)}")
            
            # Format URL directly
            log_substep(f"[red]Error extracting from {display_url}: {str(e)}[/red]")
            
            # Return a standardized error result rather than raising
            return {
                "url": url,
                "title": webpage.get("title", "Error"),
                "content": f"Error extracting content: {str(e)}",
                "success": False,
                "error": str(e),
                "score": relevance_score,
                "duration": duration,
                "cached": False,
                "is_pdf": False,
                "query": webpage.get("query", None)  # Preserve the query
            }
            
    except Exception as outer_e:
        # Handle any exceptions in the outer try block
        duration = time.time() - start_time
        logger.error(f"Fatal error in extraction process for {url if 'url' in locals() else 'unknown URL'}: {str(outer_e)}")
        
        return {
            "url": url if 'url' in locals() else "unknown-url",
            "title": "Fatal Error",
            "content": f"Fatal error in extraction process: {str(outer_e)}",
            "success": False,
            "error": str(outer_e),
            "score": webpage.get("relevance_score", 0),
            "duration": duration,
            "cached": False,
            "is_pdf": False,
            "query": webpage.get("query", None)
        }
    finally:
        # Ensure the safety timeout task is canceled
        if safety_timeout_task and not safety_timeout_task.done():
            safety_timeout_task.cancel()
            try:
                await safety_timeout_task
            except asyncio.CancelledError:
                pass  # Normal when canceling

async def extract_pdf_document(url: str) -> Dict[str, Any]:
    """Download and extract content from a PDF URL.
    
    Args:
        url: URL to PDF file
        
    Returns:
        Dictionary with content and metadata
    """
    # First try to import PDF and document loader utilities
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from core.utils.webpage import clean_html_content  # Import the cleaning utility
    except ImportError as e:
        error_msg = f"Failed to import required libraries for PDF extraction: {str(e)}"
        logger.error(error_msg)
        return {"content": error_msg, "title": "Error", "is_pdf": True}
    
    # Skip certain PDF patterns that are likely to be large directories
    filename = url.split("/")[-1].lower()
    filtered_terms = ["list", "directory", "contractors", "licensed", "registered", "members", "catalog"]
    
    if any(term in filename for term in filtered_terms):
        # This might be a large directory PDF - check first page content
        logger.warning(f"PDF filename contains directory-like terms, checking content: {filename}")
        try:
            # Try to load just the first page to see if it's a directory
            loader = PyPDFLoader(url)
            # Get just the first page for content analysis
            document = loader.load_and_split()[0]
            first_page = document.page_content
            
            # Look for signs this is a directory PDF
            is_directory = False
            
            # Check for multiple company listings
            company_indicators = re.findall(r'(?:LLC|Inc\.|Corp\.|Company|LLP|L\.L\.C\.|Co\.|Ltd\.)', first_page)
            if len(company_indicators) > 5:
                is_directory = True
                logger.warning(f"Filtered directory-style PDF with {len(company_indicators)} company indicators: {url}")
            
            # Check for tables or structured listings
            if re.search(r'(?:Name|Address|Phone|License #|Registration|ID Number|Business Name).*(?:Name|Address|Phone|License #|Registration|ID Number|Business Name)', first_page):
                is_directory = True
                logger.warning(f"Filtered directory-style PDF with table headers: {url}")
            
            # Check for numbered or bullet listings
            if len(re.findall(r'\n\d+[\.\)]\s+', first_page)) > 5:
                is_directory = True
                logger.warning(f"Filtered directory-style PDF with {len(re.findall(r'\n\d+[\.\)]\s+', first_page))} numbered listings: {url}")
            
            if is_directory:
                return {
                    "content": "PDF identified as a business directory or list document, skipped to reduce processing load.",
                    "title": f"Directory: {filename}",
                    "is_directory": True,
                    "is_pdf": True,
                    "skipped": True
                }
        except Exception as e:
            logger.error(f"Error checking PDF directory content: {str(e)}")
            # Continue with standard extraction as fallback
    
    try:
        # Load PDF using PyPDFLoader which can load from URLs
        loader = PyPDFLoader(url)
        documents = await loader.aload()
        
        # Combine all pages into a single document
        combined_content = "\n\n".join([doc.page_content for doc in documents])
        
        # Clean and format the PDF content
        logger.debug(f"Cleaning PDF content ({len(combined_content)} chars)")
        cleaned_content = combined_content
        
        # For PDFs, especially clean up the content with regex patterns
        # Remove repeated whitespace
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        # Break into paragraphs at appropriate places
        cleaned_content = re.sub(r'(\. |! |\? )', '\\1\n', cleaned_content)
        # Clean up any resulting empty lines
        cleaned_content = '\n'.join([line.strip() for line in cleaned_content.splitlines() if line.strip()])
        
        # Extract a title from the PDF content or URL
        title = ""
        # Try to extract a title from first 500 chars
        first_sentence = combined_content[:500].split('.')[0].strip()
        if 10 < len(first_sentence) < 100:  # Reasonable title length
            title = first_sentence
        else:
            # Use the filename as title
            filename = url.split('/')[-1].replace('.pdf', '').replace('-', ' ').replace('_', ' ')
            title = ' '.join([word.capitalize() for word in filename.split()])
        
        logger.debug(f"Extracted PDF with title \"{title}\" and {len(cleaned_content)} characters")
        
        return {
            "content": cleaned_content,
            "title": title,
            "is_pdf": True
        }
        
    except Exception as e:
        error_msg = f"Failed to extract PDF content from {url}: {str(e)}"
        logger.error(error_msg)
        return {"content": error_msg, "title": "PDF Extraction Error", "is_pdf": True, "error": str(e)}

# Node 4: Extract content from webpages
async def extract_webpage_content(state: BusinessOwnerSearchState) -> BusinessOwnerSearchState:
    """Extract content from selected webpages."""
    console.print(Panel(f"[bold blue]Extracting Content from Selected Webpages for[/bold blue] [yellow]{state['business_name']}[/yellow]"))
    
    if not state["webpage_scores"]:
        error_msg = f"No scored webpages to extract content from for {state['business_name']}"
        logger.warning(error_msg)
        return {
            **state, 
            "webpage_contents": [],
            "errors": state["errors"] + [error_msg]
        }
    
    start_time = time.time()
    
    # Deduplicate webpages by URL to avoid extracting the same content multiple times
    deduplicated_webpages = deduplicate_urls(state["webpage_scores"], url_field="url")
    
    # Analyze score distribution to find natural breakpoint
    scores = [webpage.get("relevance_score", 0) for webpage in deduplicated_webpages]
    avg_score = sum(scores) / len(scores) if scores else 0
    high_scoring_threshold = max(MIN_RELEVANCE_THRESHOLD, avg_score)
    
    # Separate high-scoring pages from the rest
    high_scoring_pages = [
        webpage for webpage in deduplicated_webpages 
        if webpage.get("relevance_score", 0) >= high_scoring_threshold
    ]
    remaining_pages = [
        webpage for webpage in deduplicated_webpages 
        if webpage.get("relevance_score", 0) < high_scoring_threshold
    ]
    
    # Sort remaining pages by score
    remaining_pages.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Create prioritized list with high-scoring pages first, then remaining up to limit
    max_pages = min(MAX_SEARCH_RESULTS, len(deduplicated_webpages))
    prioritized_webpages = high_scoring_pages + remaining_pages
    prioritized_webpages = prioritized_webpages[:max_pages]
    
    logger.info(f"Processing {len(prioritized_webpages)} webpages: {len(high_scoring_pages)} high-scoring (≥{high_scoring_threshold:.1f}/10) and {len(prioritized_webpages) - len(high_scoring_pages)} additional pages")
    
    # Group by domain to avoid hammering the same server with many concurrent requests
    domains = {}
    for webpage in prioritized_webpages:
        url = webpage.get("url", "")
        if not url:
            continue
            
        # Extract domain from URL
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(webpage)
        except Exception:
            # If domain extraction fails, just add to a fallback group
            if "unknown" not in domains:
                domains["unknown"] = []
            domains["unknown"].append(webpage)
    
    logger.info(f"Extracting content from {len(prioritized_webpages)} webpages across {len(domains)} domains")
    
    # Calculate concurrency limits:
    # - Global limit on total concurrent requests
    # - Per-domain limit to avoid hammering any single domain
    max_global_concurrent = MAX_CONCURRENT_EXTRACTIONS
    max_per_domain_concurrent = 2  # Max 2 concurrent requests per domain
    
    # For anti-bot sites, use even stricter limits
    anti_bot_domains = ["buzzfile.com", "manta.com", "linkedin.com", "dnb.com", "zoominfo.com"]
    anti_bot_semaphores = {domain: asyncio.Semaphore(1) for domain in anti_bot_domains}
    
    # Use a dict of semaphores to limit per-domain concurrency
    domain_semaphores = {domain: asyncio.Semaphore(max_per_domain_concurrent) for domain in domains}
    
    # Global semaphore to limit total concurrency
    global_semaphore = asyncio.Semaphore(max_global_concurrent)
    
    async def extract_with_limits(i, webpage):
        url = webpage.get("url", "")
        if not url:
            return None
            
        # Extract domain
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Check if this is an anti-bot domain
            is_anti_bot = any(abd in domain for abd in anti_bot_domains)
            
            # Select the right semaphore
            if is_anti_bot:
                # Find the matching anti-bot domain
                for abd in anti_bot_domains:
                    if abd in domain:
                        domain_sem = anti_bot_semaphores.get(abd)
                        break
                else:
                    domain_sem = domain_semaphores.get(domain)
            else:
                domain_sem = domain_semaphores.get(domain)
        except Exception:
            domain = "unknown"
            domain_sem = domain_semaphores.get("unknown")
            is_anti_bot = False
        
        # Set a timeout based on domain type and relevance score
        # Higher scoring pages get more time, anti-bot sites get more time
        relevance_score = webpage.get("relevance_score", 0)
        base_timeout = 60
        if is_anti_bot:
            # Anti-bot sites get more time, scaled by relevance
            timeout_factor = 1.5
            retries = 3
        else:
            # Regular sites get standard timeout scaled by relevance
            timeout_factor = 1.0
            retries = 2
            
        # Scale timeout by relevance score (higher score = more time)
        score_factor = min(1.0, relevance_score / 10.0 + 0.5)  # Range: 0.5 to 1.0
        final_timeout = int(base_timeout * timeout_factor * score_factor)
        
        # Update the webpage with extraction settings
        webpage["timeout"] = final_timeout
        webpage["retries"] = retries
        webpage["is_anti_bot"] = is_anti_bot
        
        # Use both global and domain-specific semaphores
        async with global_semaphore:
            # If we have a domain semaphore, use it
            if domain_sem:
                async with domain_sem:
                    return await extract_single_webpage(i, len(prioritized_webpages), webpage)
            else:
                # Fallback if no domain semaphore exists
                return await extract_single_webpage(i, len(prioritized_webpages), webpage)
    
    # Create all tasks but don't start them immediately
    extraction_tasks = []
    for i, webpage in enumerate(prioritized_webpages, 1):
        task = asyncio.create_task(extract_with_limits(i, webpage))
        extraction_tasks.append(task)
    
    # Function to log progress
    async def log_progress():
        total = len(extraction_tasks)
        while extraction_tasks:
            completed = sum(task.done() for task in extraction_tasks)
            percent = (completed / total) * 100 if total > 0 else 0
            logger.info(f"Content extraction progress: {completed}/{total} ({percent:.1f}%)")
            await asyncio.sleep(5)  # Log every 5 seconds
    
    # Start the progress logger
    progress_task = asyncio.create_task(log_progress())
    
    # Gather results, handling exceptions
    results = []
    for task in asyncio.as_completed(extraction_tasks):
        try:
            result = await task
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Extraction task failed: {str(e)}")
    
    # Cancel the progress logger
    progress_task.cancel()
    try:
        await progress_task
    except asyncio.CancelledError:
        pass
    
    # Filter successful extractions
    webpage_contents = [r for r in results if r and r.get("success", False)]
    
    # Sort by relevance score to prioritize most relevant content
    webpage_contents.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    if not webpage_contents:
        error_msg = f"Failed to extract content from any webpages for {state['business_name']}"
        logger.error(error_msg)
        return {
            **state, 
            "webpage_contents": [],
            "errors": state["errors"] + [error_msg]
        }
    
    # Log extraction statistics
    cached_count = sum(1 for result in webpage_contents if result.get("cached", False))
    total_duration = time.time() - start_time
    total_chars = sum(len(result.get("content", "")) for result in webpage_contents)
    
    logger.info(f"Extracted {len(webpage_contents)} webpages ({cached_count} from cache) in {total_duration:.2f}s, total {total_chars} characters")
    
    # Create a context for filtering based on business details
    context = f"Find information about the owner of {state['business_name']} in {state['business_state']} with zip code {state['business_zip']}"
    query = "Who owns this business? What is the residential address of the business owner?"
    
    # Import semantic filtering and smart chunking
    from core.utils.semantic_retriever import filter_relevant_content
    from core.utils.document_chunker import chunk_with_smart_size, needs_chunking, LLM_CONTEXT_THRESHOLDS
    
    # Process content with smart chunking
    chunked_contents = []
    large_doc_count = 0
    small_doc_count = 0
    
    for content_item in webpage_contents:
        # Detect if the content is from a PDF or other document type
        is_pdf = content_item.get("is_pdf", False)
        
        # Use the smart chunking helpers
        should_chunk, content_size = needs_chunking(content_item, "medium")
        
        if should_chunk:
            # Content exceeds LLM context limits, apply chunking
            logger.info(f"Smart chunking large document ({content_size} chars) from {content_item['url']} - exceeds LLM context window")
            chunks = chunk_with_smart_size(content_item)
            
            # Transfer metadata to chunks
            for i, chunk in enumerate(chunks):
                # Extract metadata from the Document object
                chunk_metadata = chunk.metadata.copy()
                chunk_dict = {
                    "url": content_item["url"],
                    "title": content_item.get("title", ""),
                    "content": chunk.page_content,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "score": content_item.get("score", 0),
                    "is_chunk": True,
                    "original_length": content_size,
                    "chunking_reason": "Exceeds LLM context window",
                    "is_pdf": is_pdf,
                    "query": content_item.get("query")  # Preserve the query
                }
                # Add any additional metadata from the chunk
                for k, v in chunk_metadata.items():
                    if k not in chunk_dict:
                        chunk_dict[k] = v
                
                chunked_contents.append(chunk_dict)
            
            logger.info(f"Split large document into {len(chunks)} chunks")
            large_doc_count += 1
        else:
            # Keep small documents as is
            content_item["is_chunk"] = False
            chunked_contents.append(content_item)
            small_doc_count += 1
    
    logger.info(f"Content preprocessing complete: {large_doc_count} large documents chunked, {small_doc_count} small documents kept intact")
    
    # Determine if semantic filtering is needed based on content volume
    total_content_size = sum(len(item.get("content", "")) for item in chunked_contents)
    total_chunk_count = len(chunked_contents)
    
    # Skip filtering for small content sets
    if total_chunk_count <= 3 and total_content_size < LLM_CONTEXT_THRESHOLDS["medium"] and large_doc_count == 0:
        logger.info(f"Skipping semantic filtering - small content set ({total_chunk_count} items, {total_content_size} chars)")
        relevant_contents = chunked_contents
    else:
        # Apply semantic filtering for larger content
        try:
            logger.info(f"Applying semantic filtering to {total_chunk_count} content items (total size: {total_content_size} chars)")
            
            # Wrap the filter_relevant_content call in a try-except to handle return format issues
            try:
                relevant_contents = await filter_relevant_content(
                    contents=chunked_contents,
                    query=query,
                    context=context,
                    similarity_threshold=0.3  # Intentionally low threshold to avoid missing information
                )
            except ValueError as ve:
                # If the function returns unexpected number of values, just use all content
                logger.error(f"Error in semantic filtering format: {str(ve)}")
                relevant_contents = chunked_contents
            
            # Ensure we got back a valid list
            if not isinstance(relevant_contents, list):
                logger.warning(f"Semantic filtering returned non-list type: {type(relevant_contents)}")
                relevant_contents = chunked_contents
            
            # Log filtering results
            logger.info(f"Filtered to {len(relevant_contents)} relevant content items from {len(chunked_contents)} items")
            
            # Calculate size reduction
            filtered_size = sum(len(item.get("content", "")) for item in relevant_contents)
            
            if total_content_size > 0:
                reduction_percentage = ((total_content_size - filtered_size) / total_content_size) * 100
                logger.info(f"Reduced content size by {reduction_percentage:.1f}% ({total_content_size} -> {filtered_size} chars)")
        except Exception as e:
            # Log the error and use all content as fallback
            logger.error(f"Error in semantic filtering: {str(e)}")
            logger.warning(f"Using all content due to filtering error")
            relevant_contents = chunked_contents
    
    # If no relevant content was found, this is an error
    if not relevant_contents:
        error_msg = f"No relevant content found for {state['business_name']}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Display results
    console.print(f"[green]✓[/green] Prepared [bold]{len(relevant_contents)}[/bold] content items for information extraction")
    
    # Update state with filtered content
    return {
        **state, 
        "webpage_contents": relevant_contents
    }

# Function to extract owners from a single webpage content
async def extract_single_owner_webpage(i: int, total: int, webpage: Dict[str, Any], chain: Any, business_name: str, business_state: str, business_zip: str) -> Dict[str, Any]:
    """Extract owner information from a single webpage."""
    start_time = time.time()
    
    # Ensure we have a valid URL
    url = "unknown-url"
    if "url" in webpage and webpage["url"]:
        url = str(webpage["url"])
    
    # Format URL for display
    display_url = url[:50] + "..." if len(url) > 50 else url
    
    logger.info(f"Analyzing webpage {i}/{total} for owner information: {url}")
    
    # Create cache key
    cache_key = f"owner_extraction_{url}_{business_name}_{business_state}_{business_zip}"
    cached_extraction = cache.get_search_results(cache_key)
    
    # We'll return a default structure if cache isn't valid
    default_extraction = {
        "url": url,
        "title": webpage.get("title", ""),
        "owners": [],
        "confidence": 0,
        "reasoning": "No owners found",
        "duration": 0
    }
    
    # Check if we have valid cached data
    if cached_extraction is not None and isinstance(cached_extraction, dict):
        # Check that we have an owners field that's a list
        if "owners" in cached_extraction and isinstance(cached_extraction["owners"], list):
            owners_field = cached_extraction["owners"]
            duration = time.time() - start_time
            owner_count = len(owners_field)
            
            # Log the cache hit
            logger.debug(f"Using cached owner extraction for {url}: {owner_count} owners")
            
            # Handle case with owners or no owners
            if owner_count > 0:
                # Build a safe owner names string
                owner_names = []
                for j in range(len(owners_field)):
                    owner_item = owners_field[j]
                    if isinstance(owner_item, dict) and "name" in owner_item:
                        owner_names.append(str(owner_item["name"]))
                
                names_str = ", ".join(owner_names) if owner_names else "unnamed owners"
                log_substep(f"Using cached extraction ({owner_count} owners) from {display_url}: {names_str}")
            else:
                log_substep(f"Using cached extraction (no owners) from {display_url}")
            
            return cached_extraction
    
    # If we got here, either no cache or invalid cache
    try:
        # Use a longer context window for extraction if content is large
        content = webpage.get("content", "")
        if not content:
            logger.warning(f"Empty content for {url}")
            raise ValueError(f"No content available for {url}")
        
        # Truncate content to prevent token limits if needed
        max_content_length = 20000
        if len(content) > max_content_length:
            content = content[:max_content_length]
            logger.debug(f"Truncated content from {len(webpage['content'])} to {max_content_length} characters")
        
        # Extract owner information directly using the LCEL chain
        # The chain should be created with a proper PydanticOutputParser
        response = await chain.ainvoke({
            "business_name": business_name,
            "business_state": business_state,
            "business_zip": business_zip,
            "webpage_content": content,
            "webpage_url": url  # Provide URL for context
        })
        
        duration = time.time() - start_time
        
        # The response is already a validated Pydantic model with owners list
        if not hasattr(response, "owners") or not response.owners:
            logger.warning(f"No owners found for {business_name} on {url}")
            log_substep(f"No owners found on {display_url}")
            return {
                "url": url,
                "title": webpage.get("title", ""),
                "owners": [],
                "duration": duration
            }
        
        # Convert Pydantic owners to plain dictionaries to make them JSON serializable
        owners_list = []
        for owner in response.owners:
            # Convert Pydantic model to dict
            if hasattr(owner, "model_dump"):
                # For Pydantic v2
                owner_dict = owner.model_dump()
            elif hasattr(owner, "dict"):
                # For Pydantic v1
                owner_dict = owner.dict()
            else:
                # Manual conversion as fallback
                owner_dict = {
                    "name": owner.name if hasattr(owner, "name") else "Unknown",
                    "confidence_score": owner.confidence_score if hasattr(owner, "confidence_score") else 0,
                    "source_url": owner.source_url if hasattr(owner, "source_url") else url,
                    "rationale": owner.rationale if hasattr(owner, "rationale") else ""
                }
            
            owners_list.append(owner_dict)
            
        # Create result dictionary
        result = {
            "url": url,
            "title": webpage.get("title", ""),
            "owners": owners_list,  # Use our plain dictionary list instead of Pydantic objects
            "duration": duration
        }
        
        # Cache the extraction result
        cache.cache_owner_extraction(business_name, business_state, business_zip, url, result)
        
        # Log extraction results
        owner_count = len(owners_list)
        if owner_count > 0:
            logger.info(f"Found {owner_count} potential owners for {business_name} on {url}")
            log_substep(f"Found {owner_count} potential owner(s) in {duration:.2f}s from {display_url}")
            
            for owner_dict in owners_list:
                log_substep(f"[green]→[/green] {owner_dict['name']} (Confidence: {owner_dict['confidence_score']}/10)")
        else:
            logger.debug(f"No owners found for {business_name} on {url}")
            log_substep(f"No owners found on {display_url}")
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error extracting owners from {url}: {str(e)}")
        log_substep(f"[red]Error extracting from {display_url}: {str(e)}[/red]")
        
        # Raise the error instead of returning a fallback
        raise ValueError(f"Error extracting owners: {str(e)}")

# Node 5: Extract owners from webpage content
async def extract_owner_information(state: BusinessOwnerSearchState) -> BusinessOwnerSearchState:
    """Extract owner information from webpage content."""
    # Import needed classes at the function level to ensure they're available
    from langchain_core.prompts import ChatPromptTemplate
    from core.configs.llm_configs import build_llm
    
    console.print(Panel(f"[bold blue]Extracting Owner Information for[/bold blue] [yellow]{state['business_name']}[/yellow]"))
    
    start_time = time.time()
    
    # Skip if no webpage content
    if not state["webpage_contents"]:
        error_msg = f"No webpage content found for {state['business_name']}. Cannot extract owners."
        logger.error(error_msg)
        return {
            **state,
            "extracted_owners": [],
            "errors": state["errors"] + [error_msg]
        }
    
    # Get the most relevant web pages (highest scores first)
    sorted_contents = sorted(
        state["webpage_contents"], 
        key=lambda x: x.get("similarity_score", 0) if x.get("similarity_score") is not None else x.get("score", 0), 
        reverse=True
    )
    
    # Cap to a maximum number of extractions
    extraction_candidates = sorted_contents[:MAX_SEARCH_RESULTS]
    
    num_candidates = len(extraction_candidates)
    logger.info(f"Extracting owner information from {num_candidates} web pages for {state['business_name']}")
    
    # Get the owner extraction prompt
    prompt = get_owner_extraction_prompt()
    llm = build_llm()
    
    # Initialize the Pydantic output parser
    parser = PydanticOutputParser(pydantic_object=BusinessOwnerList)
    format_instructions = parser.get_format_instructions()
    
    # Create a chain for extraction
    chain = prompt.partial(format_instructions=format_instructions) | llm | parser
    
    # Extract owner information from each web page asynchronously
    all_extractions = []
    
    # Create tasks for concurrent processing with semaphore to limit concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTIONS)
    
    async def wrapped_extract(i, webpage):
        async with semaphore:
            try:
                result = await extract_single_owner_webpage(
                    i, 
                    num_candidates, 
                    webpage, 
                    chain, 
                    state["business_name"], 
                    state["business_state"], 
                    state["business_zip"]
                )
                return result
            except Exception as e:
                logger.error(f"Error extracting from {webpage.get('url', 'unknown')}: {str(e)}")
                return None
    
    # Create all tasks
    tasks = [wrapped_extract(i, webpage) for i, webpage in enumerate(extraction_candidates, 1)]
    
    # Execute tasks and gather results
    results = await asyncio.gather(*tasks)
    
    # Process results - filter out None results
    for result in results:
        if result and result.get("owners"):
            all_extractions.extend(result["owners"])
    
    # Log the raw number of extractions found
    logger.info(f"Extracted {len(all_extractions)} potential owners from {num_candidates} web pages")
    
    # Handle the case where no owners were found during the first pass
    if not all_extractions:
        logger.warning(f"No owners found in first extraction pass. Using more aggressive extraction techniques for {state['business_name']}")
        
        # Create a more aggressive extraction prompt that attempts to find any potential owner information
        secondary_prompt = ChatPromptTemplate.from_template("""
You are a specialized entity extraction expert. Your task is to extract POTENTIAL business owners from the provided content.

Business Name: {business_name}
Business State: {business_state}
Business ZIP: {business_zip}

CONTENT:
{content}

IMPORTANT EXTRACTION RULES:
1. Extract ONLY fully qualified human names (with both first AND last names)
2. Do NOT extract single names, titles without names, or generic terms like "Owner" without a name
3. Look for phrases like "owned by", "founder", "president", "CEO", "manager", "principal"
4. For businesses with "LLC" or "Inc." in the name, look for registered agents or incorporation documents
5. If the business name contains a person's name (like "John Smith Plumbing"), consider that person a potential owner
6. Search for contact information that could indicate ownership
7. Exclude company names or departments as owners - only extract real human names
8. If you're uncertain if a name is complete, DO NOT include it

EXAMPLES OF VALID EXTRACTIONS:
- "John Smith" (has both first and last name)
- "Maria Rodriguez-Lopez" (complete name)
- "Dr. James Wilson" (complete name with title)

EXAMPLES OF INVALID EXTRACTIONS (DO NOT INCLUDE THESE):
- "John" (missing last name)
- "Smith" (missing first name)
- "Owner" (title without name)
- "Management Team" (not a specific person)
- "The CEO" (title without name)
- "ABC Department" (not a person)

OUTPUT FORMAT:
You MUST provide your response in valid JSON format. Output ONLY a JSON array of objects with these fields:
- "name": The person's full name (must have first AND last name)
- "confidence_score": Number from 1-10 reflecting your confidence
- "rationale": Brief explanation of why you think this person is an owner
- "source_url": "{url}"

If absolutely no valid owner names are found, return an empty JSON array [].
        """)
        
        # Try a more aggressive approach on the top 5 most relevant pages
        aggressive_tasks = []
        for i, webpage in enumerate(extraction_candidates[:5], 1):
            # Skip if no content
            if not webpage.get("content"):
                continue
                
            async def aggressive_extract(webpage):
                # Execute the aggressive prompt directly
                try:
                    # Use a special build_extraction_llm that's configured properly for JSON
                    from core.configs.llm_configs import build_extraction_llm
                    extraction_llm = build_extraction_llm(temperature=0.2)
                    
                    aggressive_result = await extraction_llm.ainvoke(
                        secondary_prompt.format(
                            business_name=state["business_name"],
                            business_state=state["business_state"],
                            business_zip=state["business_zip"],
                            content=webpage.get("content", ""),
                            url=webpage.get("url", "unknown-url")
                        )
                    )
                    
                    # Try to parse the JSON response
                    import json
                    import re
                    
                    # Find JSON in the response - ensure content is a string
                    content_str = str(aggressive_result.content) if hasattr(aggressive_result, "content") else str(aggressive_result)
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', content_str, re.DOTALL)
                    if json_match:
                        extracted_json = json_match.group(0)
                        owners_list = json.loads(extracted_json)
                        return owners_list
                    return []
                except Exception as e:
                    logger.error(f"Error in aggressive extraction: {str(e)}")
                    return []
            
            aggressive_tasks.append(aggressive_extract(webpage))
        
        # Execute aggressive tasks
        if aggressive_tasks:
            aggressive_results = await asyncio.gather(*aggressive_tasks)
            
            # Flatten results
            for owners_list in aggressive_results:
                if owners_list:
                    all_extractions.extend(owners_list)
        
        # If still no owners, try to extract from business name
        if not all_extractions:
            # Check if business name contains a person's name
            from core.configs.llm_configs import build_llm
            name_extraction_llm = build_llm(temperature=0, response_format_type=None)  # Don't use JSON format for this simple question
            
            name_extraction_prompt = ChatPromptTemplate.from_template(
                """Does the business name '{business_name}' contain a FULL person's name (with both first AND last name)?
                
                If yes, extract ONLY that person's full name (first AND last name).
                If no complete person's name is present, respond with ONLY 'No complete name found'.
                
                Examples:
                - "Smith Plumbing LLC" -> "No complete name found" (missing first name)
                - "John Smith Plumbing" -> "John Smith"
                - "ABC Construction" -> "No complete name found"
                - "Johnson & Sons Electrical" -> "No complete name found" (missing first name)
                - "Robert Johnson Electric" -> "Robert Johnson"
                
                Do NOT extract partial names - only extract if BOTH first and last names are present.
                """
            )
            
            name_in_business = await name_extraction_llm.ainvoke(
                name_extraction_prompt.format(business_name=state["business_name"])
            )
            
            # Ensure content is a string before calling strip()
            name_result = str(name_in_business.content).strip() if hasattr(name_in_business, "content") else str(name_in_business).strip()
            
            # Validate that we have a real person name, not just "No"
            if name_result and not name_result.lower().startswith("no "):
                # Verify we have a proper full name (contains at least two words, not just "No complete name found")
                name_parts = name_result.split()
                if len(name_parts) >= 2 and not name_result.lower().startswith("no complete"):
                    all_extractions.append({
                        "name": name_result,
                        "confidence_score": 5,
                        "rationale": "Extracted from business name",
                        "source_url": "Business name analysis"
                    })
    
    # Remove duplicates by owner name
    unique_owners = {}
    for owner in all_extractions:
        name = owner.get("name", "").strip()
        # Skip invalid or empty names
        if not name or name.lower() in ["unknown", "n/a", "none", "no", "no."]:
            continue
            
        # Validate that name contains at least two words (first and last name)
        name_parts = name.split()
        if len(name_parts) < 2:
            logger.warning(f"Skipping incomplete name: {name}")
            continue
            
        # If we already have this owner, keep the one with higher confidence
        if name in unique_owners:
            existing = unique_owners[name]
            if owner.get("confidence_score", 0) > existing.get("confidence_score", 0):
                unique_owners[name] = owner
        else:
            unique_owners[name] = owner
    
    final_extractions = list(unique_owners.values())
    
    # Sort by confidence score (highest first)
    final_extractions.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)
    
    # Log the owners found
    logger.info(f"Found {len(final_extractions)} unique owners for {state['business_name']}")
    for i, owner in enumerate(final_extractions[:10], 1):  # Log only top 10
        log_substep(f"Owner {i}: {owner.get('name', 'Unknown')} (Confidence: {owner.get('confidence_score', 0)}/10)")
    
    # Calculate duration
    duration = time.time() - start_time
    logger.debug(f"Owner extraction took {duration:.2f} seconds")
    
    # Display summary
    console.print(f"[green]✓[/green] Extracted [bold]{len(final_extractions)}[/bold] potential owner(s) in [bold]{duration:.2f}s[/bold]")
    
    # Update state with extracted owners
    return {
        **state,
        "extracted_owners": final_extractions
    }

# Node 6: Synthesize final owner list
async def synthesize_final_owners(state: BusinessOwnerSearchState) -> BusinessOwnerSearchState:
    """Synthesize final owner information from all extracted data."""
    # Import ChatPromptTemplate at the function level to ensure it's available
    from langchain_core.prompts import ChatPromptTemplate
    
    console.print(Panel(f"[bold blue]Synthesizing Final Results for[/bold blue] [yellow]{state['business_name']}[/yellow]"))
    
    start_time = time.time()
    
    # If no owners were extracted, return empty list but not error
    if not state["extracted_owners"]:
        message = f"No owner candidates were found for {state['business_name']}. No further synthesis needed."
        logger.warning(message)
        console.print(f"[yellow]⚠[/yellow] {message}")
        
        return {
            **state,
            "final_owners": []
        }
    
    # Filter out low-confidence owners and incomplete/invalid names
    MIN_CONFIDENCE_FOR_SYNTHESIS = 4  # Higher threshold to improve quality of results
    extracted_owners = []
    
    for owner in state["extracted_owners"]:
        # Skip low confidence owners
        if owner.get("confidence_score", 0) < MIN_CONFIDENCE_FOR_SYNTHESIS:
            continue
            
        # Skip invalid names (blanks, single names, etc.)
        name = owner.get("name", "").strip()
        if not name or name.lower() in ["no", "no.", "none", "n/a", "unknown", "not found"]:
            continue
            
        # Make sure we have both first and last names
        name_parts = name.split()
        if len(name_parts) < 2:
            logger.warning(f"Skipping incomplete name in synthesis: {name}")
            continue
            
        # Add to filtered list
        extracted_owners.append(owner)
    
    # If all owners are below threshold, use only the top highest confidence ones above a minimum bar
    if not extracted_owners:
        logger.warning(f"All extracted owner candidates for {state['business_name']} had confidence scores below {MIN_CONFIDENCE_FOR_SYNTHESIS} or incomplete names.")
        console.print(f"[yellow]⚠[/yellow] All extracted owners had low confidence scores or incomplete names. Using highest confidence candidates anyway.")
        
        # Filter out any obviously wrong names like "No." and ensure complete names
        filtered_candidates = []
        
        for owner in state["extracted_owners"]:
            if owner.get("confidence_score", 0) < 3:  # Minimum score of 3
                continue
                
            # Validate the name format
            name = owner.get("name", "").strip()
            if not name or name.lower() in ["no", "no.", "none", "n/a", "unknown", "not found"]:
                continue
                
            # Make sure we have both first and last names
            name_parts = name.split()
            if len(name_parts) < 2:
                continue
                
            filtered_candidates.append(owner)
        
        # Sort by confidence and take top 2 if we have any
        if filtered_candidates:
            extracted_owners = sorted(
                filtered_candidates,
                key=lambda x: x.get("confidence_score", 0),
                reverse=True
            )[:2]  # Only take top 2 highest confidence owners as fallback
        else:
            # If no decent candidates, return empty
            logger.warning(f"No viable owner candidates for {state['business_name']} after filtering.")
            return {
                **state,
                "final_owners": []
            }
    
    total_candidates = len(extracted_owners)
    logger.info(f"Synthesizing final results from {total_candidates} owner candidates for {state['business_name']}")
    
    # Try to synthesize final results
    try:
        # Format the extracted owner information for the synthesis prompt
        owner_information = "\n\n".join([
            f"Source: {owner.get('source_url', 'Unknown')}\n"
            f"Name: {owner.get('name', 'Unknown')}\n"
            f"Role: {owner.get('role', 'Unknown')}\n"
            f"Address: {owner.get('address', 'Unknown')}\n"
            f"Confidence: {owner.get('confidence_score', 0)}/10\n"
            f"Rationale: {owner.get('rationale', 'No rationale provided')}"
            for owner in extracted_owners
        ])
        
        # Get the final result synthesis prompt
        prompt = get_final_result_synthesis_prompt()
        llm = build_llm()
        
        # Initialize the Pydantic output parser
        parser = PydanticOutputParser(pydantic_object=BusinessOwnerList)
        format_instructions = parser.get_format_instructions()
        
        # Create a chain for the synthesis
        chain = prompt.partial(format_instructions=format_instructions) | llm | parser
        
        # Add instructions to help with synthesis
        instructions = """CRITICAL RULES:
        1. Use ALL available evidence to make your determination
        2. Only include owners with strong supporting evidence
        3. Only include owners with COMPLETE names (both first AND last name)
        4. For small businesses, the owner's full name may appear in the business name itself
        5. Look for consistent information across multiple sources
        6. VERIFY that owners are explicitly connected to the specific business
        7. NEVER make up or hallucinate owner information not in the provided data
        8. Exclude responses like "No", "No.", "None", "N/A", or "Unknown" as actual names
        9. If NO OWNERS can be confidently identified, return an empty list for owners
        10. Only include owners with confidence score of 6 or higher unless there's compelling evidence
        11. DO NOT include generic titles, abbreviations, or single names as owners
        """
        
        # Run the synthesis
        response = await chain.ainvoke({
            "business_name": state["business_name"],
            "business_state": state["business_state"],
            "business_zip": state["business_zip"],
            "owner_information": owner_information + "\n\n" + instructions
        })
        
        # Process the output
        # If owners list is empty, we'll use a fallback approach
        if not hasattr(response, "owners") or not response.owners:
            logger.warning(f"No final owners identified for {state['business_name']} after synthesis")
            console.print(f"[yellow]⚠[/yellow] No owners identified with sufficient confidence.")
            
            # FALLBACK: If synthesis fails, take the highest confidence extracted owners directly
            fallback_owners = sorted(
                extracted_owners, 
                key=lambda x: x.get("confidence_score", 0), 
                reverse=True
            )[:2]  # Take top 2 highest confidence owners max
            
            # Only use fallback if we have at least one owner with reasonable confidence
            if fallback_owners and fallback_owners[0].get("confidence_score", 0) >= 5:
                logger.info(f"Using fallback approach: Using top {len(fallback_owners)} highest confidence extracted owners directly")
                
                # Create final owners list
                final_owners = fallback_owners
                
                # Display summary
                owner_names = ", ".join([owner.get("name", "Unknown") for owner in final_owners])
                console.print(f"[yellow]⚠[/yellow] Using fallback approach: [bold]{len(final_owners)}[/bold] owner(s) from direct extraction: [yellow]{owner_names}[/yellow]")
                
                # Update state with final owners
                return {
                    **state,
                    "final_owners": final_owners
                }
            
            # If fallback didn't work either, return empty list without error
            return {
                **state,
                "final_owners": []
            }
        
        # Convert the owners to dictionaries and perform validation
        final_owners = []
        
        if hasattr(response, "owners"):
            for owner in response.owners:
                # Convert to dict format
                if hasattr(owner, "model_dump"):
                    owner_dict = owner.model_dump()
                elif hasattr(owner, "dict"):
                    owner_dict = owner.dict()
                else:
                    owner_dict = {
                        "name": owner.name if hasattr(owner, "name") else "Unknown",
                        "address": owner.address if hasattr(owner, "address") else None,
                        "confidence_score": owner.confidence_score if hasattr(owner, "confidence_score") else 0,
                        "source_url": owner.source_url if hasattr(owner, "source_url") else "",
                        "rationale": owner.rationale if hasattr(owner, "rationale") else "No rationale provided"
                    }
                
                # Validate owner name - skip invalid names
                owner_name = owner_dict.get("name", "").strip()
                if (not owner_name or 
                    owner_name.lower() in ["no", "no.", "none", "n/a", "unknown", "not found"] or
                    len(owner_name) < 3):
                    logger.warning(f"Skipping invalid owner name: {owner_name}")
                    continue
                
                # Ensure name contains both first and last name
                name_parts = owner_name.split()
                if len(name_parts) < 2:
                    logger.warning(f"Skipping owner with incomplete name (missing first or last name): {owner_name}")
                    continue
                    
                # Set minimum confidence to 5 for any result to avoid low-quality data
                if owner_dict.get("confidence_score", 0) < 5:
                    owner_dict["confidence_score"] = 5
                
                final_owners.append(owner_dict)
        
        # Log the results
        if final_owners:
            logger.info(f"Identified {len(final_owners)} final owners for {state['business_name']}")
            for i, owner in enumerate(final_owners, 1):
                log_substep(f"Owner {i}: {owner.get('name', 'Unknown')} (Confidence: {owner.get('confidence_score', 0)}/10)")
        else:
            logger.warning(f"No final owners identified for {state['business_name']} after synthesis")
        
        # Calculate duration
        duration = time.time() - start_time
        logger.debug(f"Owner synthesis took {duration:.2f} seconds")
        
        # Display summary
        if final_owners:
            owner_names = ", ".join([owner.get("name", "Unknown") for owner in final_owners])
            console.print(f"[green]✓[/green] Identified [bold]{len(final_owners)}[/bold] owner(s): [yellow]{owner_names}[/yellow]")
        else:
            console.print(f"[yellow]⚠[/yellow] No owners identified with sufficient confidence.")
        
        # Update state with final owners
        return {
            **state,
            "final_owners": final_owners
        }
    except Exception as e:
        # Log the error but don't raise - return empty list
        error_msg = f"Error synthesizing owner information: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        
        # Try to gracefully recover with fallback
        try:
            # FALLBACK: Use highest confidence extracted owners directly if synthesis fails
            fallback_owners = sorted(
                extracted_owners, 
                key=lambda x: x.get("confidence_score", 0), 
                reverse=True
            )[:2]  # Take top 2 highest confidence owners max
            
            # Only use fallback if we have at least one owner with reasonable confidence
            if fallback_owners and fallback_owners[0].get("confidence_score", 0) >= 5:
                logger.info(f"Using fallback after error: Top {len(fallback_owners)} highest confidence extracted owners")
                return {
                    **state,
                    "final_owners": fallback_owners,
                    "errors": state["errors"] + [error_msg]
                }
        except Exception:
            # If even fallback fails, return empty list
            pass
            
        return {
            **state,
            "final_owners": [],
            "errors": state["errors"] + [error_msg]
        }

# Create the business owner search graph
def create_business_owner_search_graph() -> StateGraph:
    """Create the LangGraph workflow for business owner search."""
    # Initialize graph
    workflow = StateGraph(BusinessOwnerSearchState)
    
    # Add nodes
    workflow.add_node("generate_search_queries", generate_search_queries)
    workflow.add_node("execute_search_queries", execute_search_queries)
    workflow.add_node("score_webpage_relevance", score_webpage_relevance)
    workflow.add_node("extract_webpage_content", extract_webpage_content)
    workflow.add_node("extract_owner_information", extract_owner_information)
    workflow.add_node("synthesize_final_owners", synthesize_final_owners)
    
    # Define edges
    workflow.add_edge("generate_search_queries", "execute_search_queries")
    workflow.add_edge("execute_search_queries", "score_webpage_relevance")
    workflow.add_edge("score_webpage_relevance", "extract_webpage_content")
    workflow.add_edge("extract_webpage_content", "extract_owner_information")
    workflow.add_edge("extract_owner_information", "synthesize_final_owners")
    workflow.add_edge("synthesize_final_owners", END)
    
    # Set entry point
    workflow.set_entry_point("generate_search_queries")
    
    return workflow 