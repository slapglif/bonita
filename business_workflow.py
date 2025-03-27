"""
Business Information Extraction Workflow

This module implements an advanced async workflow to extract business owner information
using multi-query generation, content crawling, RAG and reflection for maximum accuracy.
The workflow uses LangMem to store and retrieve semantic information between steps.
"""

import os
import asyncio
import time
import traceback
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from datetime import datetime

try:
    # For Pydantic v2
    from pydantic import BaseModel, Field
except ImportError:
    # For older Pydantic versions
    from pydantic.v1 import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

# Rich and Loguru for enhanced logging and observability
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.traceback import install as install_rich_traceback
from loguru import logger

from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END
from langgraph.store.memory import InMemoryStore

# Configure Rich and Loguru for enhanced observability
console = Console(record=True)
install_rich_traceback(show_locals=True)

# Setup custom progress display
progress = Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TextColumn("[bold green]{task.completed} of {task.total}"),
    TimeElapsedColumn(),
)


def get_langmem_store():
    """Get a memory store instance from LangGraph."""
    store = InMemoryStore(
        index={
            "dims": 1536,  # Dimension for text-embedding-3-small
            "embed":
            "openai:text-embedding-3-small",  # Specify the embedding model
        })
    return store


# Remove standard logging and configure Loguru
import logging

logging.getLogger().handlers = []
logger.remove()  # Remove default handler
logger.add(
    lambda msg: console.print(f"[bold blue]{msg}[/bold blue]"),
    level="INFO",
    format=
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)
logger.add("workflow.log", rotation="10 MB", retention="1 day", level="DEBUG")


class BusinessQuery(BaseModel):
    """A structured query format for SerpAPI searches."""

    business_name: str = Field(..., description="The full business name")
    location: str = Field(
        ..., description="Business location including state and zip code")
    query_type: str = Field(
        ..., description="Type of query: 'owner', 'address', 'contact'")
    search_query: str = Field(
        ..., description="The full search query to submit to search engine")

    class Config:
        schema_extra = {
            "example": {
                "business_name":
                "Absolute Home Improvement LLC",
                "location":
                "MO 64040",
                "query_type":
                "owner",
                "search_query":
                "who owns Absolute Home Improvement LLC in Missouri 64040 owner name",
            }
        }


class SearchResult(BaseModel):
    """Structure for storing search results with metadata."""

    business_name: str
    query_type: str
    search_url: str
    snippet: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)

    class Config:
        schema_extra = {
            "example": {
                "business_name": "Absolute Home Improvement LLC",
                "query_type": "owner",
                "search_url": "https://example.com/business/info",
                "snippet":
                "Absolute Home Improvement LLC is owned by John Smith who started the company in 2015.",
                "relevance_score": 0.85,
            }
        }


class ContentChunk(BaseModel):
    """Represents a processed content chunk from a webpage."""

    business_name: str
    source_url: str
    content: str
    chunk_index: int
    embedding_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "business_name": "Absolute Home Improvement LLC",
                "source_url": "https://example.com/business/info",
                "content":
                "John Smith is the founder and owner of Absolute Home Improvement LLC.",
                "chunk_index": 1,
                "embedding_id": "emb_12345",
            }
        }


class BusinessOwnerInfo(BaseModel):
    """Final extracted business owner information with analysis insights."""

    business_name: str
    owner_name: Optional[str] = Field(
        None, description="Full name of the business owner")
    primary_address: Optional[str] = Field(
        None, description="Primary business address")
    state: str
    zip_code: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)
    analysis_notes: Optional[str] = Field(
        None, description="Detailed analysis of information sources and reasoning process")
    
    # Additional fields for improved information tracking
    owner_confidence: float = Field(0.0, ge=0.0, le=1.0, 
        description="Confidence specifically for owner information")
    address_confidence: float = Field(0.0, ge=0.0, le=1.0, 
        description="Confidence specifically for address information")
    alternative_owners: List[str] = Field(default_factory=list,
        description="Other potential owners if multiple possibilities exist")

    class Config:
        schema_extra = {
            "example": {
                "business_name":
                "Absolute Home Improvement LLC",
                "owner_name":
                "John Smith",
                "primary_address":
                "123 Main Street, Kansas City",
                "state":
                "MO",
                "zip_code":
                "64040",
                "confidence_score":
                0.92,
                "sources": [
                    "https://example.com/business/info",
                    "https://example.com/directory",
                ],
            }
        }


class QueryGenerator(BaseModel):
    """Memory item to store generated queries for each business."""

    business_name: str
    queries: List[BusinessQuery] = Field(default_factory=list)
    reflection_notes: Optional[str] = None


class ProcessingMemory(BaseModel):
    """Memory item to track processing state for a business."""

    business_name: str
    state: str
    zip_code: str
    search_results: List[SearchResult] = Field(default_factory=list)
    content_chunks: List[ContentChunk] = Field(default_factory=list)
    processing_stage: str = "initial"
    owner_name_candidates: List[Tuple[str,
                                      float]] = Field(default_factory=list)
    address_candidates: List[Tuple[str, float]] = Field(default_factory=list)
    reflection_notes: Optional[str] = None


async def generate_search_queries(
        llm: ChatOpenAI, business_data: Dict[str, Any]) -> List[BusinessQuery]:
    """Generate multiple search queries for different aspects of business information.
    
    This enhanced version casts a wider net by:
    1. Generating more query variations per category
    2. Including industry-specific and business-type-specific queries
    3. Adding location-based diversification
    4. Using more advanced search operators for precise results
    """
    # Extract business information
    business_name = business_data.get("business_name", "")
    state = business_data.get("state", "")
    zip_code = business_data.get("zip_code", "")
    location = f"{state} {zip_code}"

    # Create the prompt with enhanced search strategy
    query_generation_prompt = ChatPromptTemplate.from_template("""
    You are an EXPERT business intelligence researcher specializing in discovering hard-to-find information about businesses.
    Your task is to cast the WIDEST POSSIBLE NET when generating search queries - we need to maximize information discovery.
    
    I need comprehensive research on this business:
    Business Name: {business_name}
    Location: {location}
    
    Generate multiple diverse search queries (at least 10 total) across these categories:
    1. OWNER INFORMATION (5+ queries): 
       - Direct owner searches using different name formats and terms (CEO, founder, president, principal)
       - Secretary of State business filings and registrations
       - Industry-specific directories and trade associations
       - Business network sites like LinkedIn
       - News articles about the business
       
    2. ADDRESS INFORMATION (3+ queries):
       - Physical location and headquarters searches
       - State business registration address records
       - Multiple variations of location queries
       
    3. BUSINESS VERIFICATION (2+ queries):
       - Contact information for verification
       - Business license and registration numbers
       
    FORMAT REQUIREMENTS:
    Return your response as a JSON array with each query containing:
    - business_name: "{business_name}"
    - location: "{location}"
    - query_type: One of ["owner", "address", "registration", "contact", "industry"]
    - search_query: The exact query text to run, using advanced operators when helpful
    
    Make each search query HIGHLY DIVERSE - use different search techniques, advanced operators, 
    and varied approaches for maximum coverage.
    
    DIVERSITY REQUIREMENTS:
    - Use different search syntaxes (quotes, site: operator, etc.)
    - Target different source types (official records, social media, news, directories)
    - Use different word orders and synonyms
    - Incorporate both exact phrase matches and broader contextual searches
    
    IMPORTANT: The React agent will utilize these search results to decide where to expand investigations.
    """)

    # Parse output as JSON
    output_parser = JsonOutputParser()

    # Create the chain
    query_chain = query_generation_prompt | llm | output_parser

    # Generate queries
    try:
        queries_data = await query_chain.ainvoke({
            "business_name": business_name,
            "location": location
        })

        # Convert to BusinessQuery objects
        queries = []
        for query_data in queries_data:
            query = BusinessQuery(
                business_name=business_name,
                location=location,
                query_type=query_data.get("query_type", "general"),
                search_query=query_data.get("search_query", ""),
            )
            queries.append(query)

        return queries

    except Exception as e:
        logger.error(f"Error generating queries for {business_name}: {str(e)}")
        # Return a default query as fallback
        return [
            BusinessQuery(
                business_name=business_name,
                location=location,
                query_type="owner",
                search_query=
                f"who owns {business_name} in {location} owner name",
            ),
            BusinessQuery(
                business_name=business_name,
                location=location,
                query_type="address",
                search_query=
                f"{business_name} address {location} headquarters location",
            ),
        ]


@retry(stop=stop_after_attempt(2),
       wait=wait_exponential(multiplier=1, min=2, max=5))
async def search_with_query(query: BusinessQuery) -> List[SearchResult]:
    """Perform search using SerpAPI with efficient error handling and retry logic.
    
    This implementation maximizes parallel processing capability by using SerpAPI directly
    with minimal overhead.
    """
    start_time = time.time()
    try:
        # Log the search operation with minimal overhead
        logger.info(
            f"Executing search query for: {query.business_name} ({query.query_type})"
        )

        # Verify API key without any UI overhead
        serpapi_key = os.environ.get("SERPAPI_API_KEY")
        if not serpapi_key:
            logger.error("SERPAPI_API_KEY environment variable not set")
            return []  # Return empty results without retry to avoid hanging

        # Initialize and execute SerpAPI call directly
        search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
        search_result = await search.aresults(
            query.search_query)  # Use async version for better performance

        # Process results directly without complex UI operations
        results = []
        if isinstance(search_result,
                      dict) and "organic_results" in search_result:
            # Take top results (up to 10 for maximum data gathering)
            for i, result in enumerate(search_result["organic_results"][:10]):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")

                if not link:  # Try to find URL in other fields if link is missing
                    link = result.get("displayed_link",
                                      result.get("displayLink", ""))

                # Fast relevance scoring
                position_score = 1.0 - (i * 0.1)  # Position-based score

                # Simple keyword matching for relevance without complex processing
                relevance_boost = 0.0
                search_text = (title + " " + snippet).lower()

                # Business name match
                if query.business_name.lower() in search_text:
                    relevance_boost += 0.3

                # Query type keywords
                query_type_terms = {
                    "owner":
                    ["owner", "founder", "ceo", "president", "director"],
                    "address":
                    ["address", "location", "headquarters", "office", "based"],
                    "contact":
                    ["contact", "phone", "email", "website", "reach"]
                }

                # Check for query type keyword matches
                for keyword in query_type_terms.get(query.query_type, []):
                    if keyword in search_text:
                        relevance_boost += 0.1
                        break  # Only count once per category

                # Calculate final score with bounds check
                relevance_score = min(
                    1.0, max(0.1, position_score + relevance_boost))

                # Create result object
                search_result_obj = SearchResult(
                    business_name=query.business_name,
                    query_type=query.query_type,
                    search_url=link,
                    snippet=snippet,
                    relevance_score=relevance_score,
                )
                results.append(search_result_obj)

            # Log completion without UI overhead
            elapsed = time.time() - start_time
            logger.info(
                f"Found {len(results)} results for {query.business_name} in {elapsed:.2f}s"
            )
        else:
            logger.warning(
                f"No organic results found for query: {query.search_query}")

        return results

    except Exception as e:
        # Log error without UI overhead
        elapsed = time.time() - start_time
        logger.error(
            f"Search error for {query.business_name} ({elapsed:.2f}s): {str(e)}"
        )
        # Return empty results to allow processing to continue with other queries
        return []


async def fetch_content(urls: List[str]) -> Dict[str, str]:
    """Fetch and extract content from URLs using LangChain's PlaywrightURLLoader with maximum parallelism."""
    content_map = {}

    if not urls:
        return content_map

    try:
        # Import here to catch ImportError early - only use LangChain's wrapper
        from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader
    except ImportError:
        logger.error("PlaywrightURLLoader not found. Cannot fetch content.")
        # Return empty content for all URLs
        return {url: f"Failed to load content for {url}" for url in urls}

    try:
        # LangChain's PlaywrightURLLoader already has built-in parallelism capabilities
        logger.info(
            f"Fetching content from {len(urls)} URLs using PlaywrightURLLoader"
        )

        # Configure loader with optimal performance settings
        loader = PlaywrightURLLoader(urls=urls,
                                     remove_selectors=[
                                         "nav", "header", "footer", ".ads",
                                         "#ads", ".navigation", ".menu"
                                     ],
                                     headless=True)

        # Load documents (this is already efficient and runs in parallel)
        docs = loader.load()
        logger.info(f"Successfully loaded {len(docs)} documents")

        # Map documents to our content dictionary
        for doc in docs:
            url = doc.metadata.get("source")
            if url:
                content_map[url] = doc.page_content

        # Add placeholder entries for any URLs that failed to load
        for url in urls:
            if url not in content_map:
                content_map[url] = f"Failed to load content for {url}"

    except Exception as e:
        logger.error(f"Error fetching content: {str(e)}")
        # Add error message content for all URLs
        for url in urls:
            if url not in content_map:
                content_map[url] = f"Failed to load content: {str(e)}"

    # Ensure all URLs have an entry
    for url in urls:
        if url not in content_map:
            content_map[url] = "No content retrieved"

    logger.info(f"Fetched content from {len(content_map)} URLs")
    return content_map


async def preprocess_content(content_map: Dict[str, str],
                             business_name: str) -> List[ContentChunk]:
    """Split and preprocess content into chunks for embedding."""
    chunks = []

    for url, content in content_map.items():
        if not content:
            continue

        # Simple chunking by paragraphs
        paragraphs = [p for p in content.split("\n\n") if p.strip()]

        for i, paragraph in enumerate(paragraphs):
            # Only keep chunks with some minimum content
            if len(paragraph.strip()) < 50:
                continue

            chunk = ContentChunk(
                business_name=business_name,
                source_url=url,
                content=paragraph.strip(),
                chunk_index=i,
            )
            chunks.append(chunk)

    return chunks


async def score_and_rank_chunks(
        chunks: List[ContentChunk], query: str,
        embeddings: Embeddings) -> List[Tuple[ContentChunk, float]]:
    """Score and rank content chunks based on relevance to query."""
    if not chunks:
        return []

    try:
        # Create documents from chunks
        docs = [
            Document(page_content=chunk.content, metadata={"index": i})
            for i, chunk in enumerate(chunks)
        ]

        # Get query embedding
        query_embedding = embeddings.embed_query(query)

        # Get document embeddings
        doc_embeddings = embeddings.embed_documents(
            [doc.page_content for doc in docs])

        # Calculate similarity scores
        scored_chunks = []
        for i, doc_embedding in enumerate(doc_embeddings):
            # Calculate cosine similarity
            similarity = sum(
                a * b for a, b in zip(query_embedding, doc_embedding)) / (
                    sum(a * a for a in query_embedding)**0.5 *
                    sum(b * b for b in doc_embedding)**0.5)

            chunk = chunks[i]
            scored_chunks.append((chunk, similarity))

        # Sort by score in descending order
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return scored_chunks

    except Exception as e:
        logger.error(f"Error scoring chunks: {str(e)}")
        # Return unsorted chunks with neutral scores as fallback
        return [(chunk, 0.5) for chunk in chunks]


async def extract_business_info(llm: ChatOpenAI, business_data: Dict[str, Any],
                                memory: ProcessingMemory) -> BusinessOwnerInfo:
    """Extract structured business owner information from processed data with advanced analysis.
    
    This enhanced version implements a React-style thinking approach to extract information by:
    1. Analyzing all available evidence systematically
    2. Identifying contradictions or confirmations across multiple sources
    3. Weighing source credibility differently (e.g., official records > social media)
    4. Using multiple reasoning steps before reaching conclusions
    5. Identifying patterns across different source types
    """
    # Format all the content chunks and search results
    chunk_texts = []
    for chunk in memory.content_chunks:
        chunk_texts.append(
            f"SOURCE: {chunk.source_url}\nCONTENT: {chunk.content}")

    search_texts = []
    for result in memory.search_results:
        search_texts.append(
            f"QUERY TYPE: {result.query_type}\nURL: {result.search_url}\nSNIPPET: {result.snippet}"
        )

    # Create the enhanced prompt with React-style reasoning
    extraction_prompt = ChatPromptTemplate.from_template("""
    You are an expert business intelligence analyst who uses systematic reasoning to extract verified information.
    
    BUSINESS INFORMATION:
    Business Name: {business_name}
    State: {state}
    Zip Code: {zip_code}
    
    SEARCH RESULTS:
    {search_results}
    
    CONTENT CHUNKS:
    {content_chunks}
    
    TASK:
    Using a React-style multi-step analysis process with explicit reasoning, extract the following business details:
    
    1. Business Owner's Full Name(s)
    2. Primary Business Address
    
    ANALYSIS PROCESS - Follow this structured approach:
    
    Step 1: SOURCE EVALUATION
    - Categorize each source by type (government record, business directory, social media, news, etc.)
    - Rank sources by likely reliability (government > business registries > news > social > general web)
    - Identify which sources are most authoritative for each information type
    
    Step 2: EVIDENCE COLLECTION
    - Identify all explicit mentions of owners, executives, founders, or principals
    - Collect all address information, noting which appear to be primary business locations
    - Note any contradictory information across sources
    - List alternative candidates when multiple possibilities exist
    
    Step 3: PATTERN RECOGNITION
    - Look for repeated mentions of the same person/address across multiple independent sources
    - Identify patterns that suggest official vs. subsidiary relationships
    - Analyze consistency of information across different source types
    
    Step 4: CONFIDENCE ASSESSMENT
    - Evaluate confidence based on source reliability, consistency across sources, specificity of claims
    - Calculate separate confidence scores for owner and address information
    - Explicitly note when evidence is insufficient or contradictory
    - Document your reasoning about confidence levels
    
    Step 5: EXPANSION DECISION
    - Determine if additional research would be beneficial
    - Note specific areas where more information would increase confidence
    
    If specific information is not clearly stated or cannot be reliably determined, indicate with "Not found in sources".
    
    Respond in JSON format:
    {{
        "business_name": "{business_name}",
        "owner_name": "Extracted owner name or 'Not found in sources'",
        "primary_address": "Extracted address or 'Not found in sources'",
        "state": "{state}",
        "zip_code": "{zip_code}",
        "confidence_score": 0.95,
        "sources": ["list", "of", "source", "urls", "that", "provided", "the", "information"],
        "analysis_notes": "Detailed analysis of your reasoning process, source evaluation, and evidence assessment",
        "owner_confidence": 0.85,
        "address_confidence": 0.90,
        "alternative_owners": ["List", "of", "alternative", "owner", "candidates"],
        "needs_expansion": true/false,
        "expansion_rationale": "Explanation of why additional search would be helpful or unnecessary"
    }}
    """)

    # Parse output as JSON
    output_parser = JsonOutputParser()

    # Create the chain
    extraction_chain = extraction_prompt | llm | output_parser

    # Extract information
    try:
        result = await extraction_chain.ainvoke({
            "business_name":
            business_data.get("business_name", ""),
            "state":
            business_data.get("state", ""),
            "zip_code":
            business_data.get("zip_code", ""),
            "search_results":
            "\n\n".join(search_texts),
            "content_chunks":
            "\n\n".join(chunk_texts),
        })

        # Convert to enhanced BusinessOwnerInfo object with detailed analysis
        # Evaluate need for search expansion based on explicit agent output
        needs_expansion = result.get("needs_expansion", False)
        expansion_rationale = result.get("expansion_rationale", "")
        
        # Add to analysis notes if search expansion is needed
        analysis_notes = result.get("analysis_notes", "No detailed analysis available")
        if needs_expansion:
            analysis_notes += f"\n\nAUTOMATIC SEARCH EXPANSION RECOMMENDED:\n{expansion_rationale}"
            
        # Check confidence thresholds to determine if we need more information
        owner_confidence = float(result.get("owner_confidence", result.get("confidence_score", 0.0)))
        address_confidence = float(result.get("address_confidence", result.get("confidence_score", 0.0)))
        
        # If confidence is too low but not explicitly marked for expansion, add recommendation
        if (owner_confidence < 0.6 or address_confidence < 0.6) and not needs_expansion:
            analysis_notes += "\n\nLOW CONFIDENCE DETECTED: Additional search strategies may improve results."
            
        # Create the business owner info with enhanced analysis
        business_info = BusinessOwnerInfo(
            business_name=result.get("business_name",
                                     business_data.get("business_name", "")),
            owner_name=result.get("owner_name", "Not found in sources"),
            primary_address=result.get("primary_address",
                                       "Not found in sources"),
            state=result.get("state", business_data.get("state", "")),
            zip_code=result.get("zip_code", business_data.get("zip_code", "")),
            confidence_score=float(result.get("confidence_score", 0.0)),
            sources=result.get("sources", []),
            # Add the enhanced analysis fields
            analysis_notes=analysis_notes,
            owner_confidence=owner_confidence,
            address_confidence=address_confidence,
            alternative_owners=result.get("alternative_owners", []),
        )

        return business_info

    except Exception as e:
        logger.error(f"Error extracting business info: {str(e)}")
        # Return default object with error indication and analysis notes
        return BusinessOwnerInfo(
            business_name=business_data.get("business_name", ""),
            owner_name="Error during extraction",
            primary_address="Error during extraction",
            state=business_data.get("state", ""),
            zip_code=business_data.get("zip_code", ""),
            confidence_score=0.0,
            sources=[],
            analysis_notes=f"Error during extraction: {str(e)}\n{traceback.format_exc()}",
            owner_confidence=0.0,
            address_confidence=0.0,
            alternative_owners=[],
        )


class BusinessInfoExtractor:
    """Main class to orchestrate the business information extraction workflow."""

    def __init__(self):
        """Initialize the business information extractor."""
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        # Initialize memory store
        # Always use LangGraph's InMemoryStore
        self.store = get_langmem_store()
        logger.info("Initialized memory store for business extraction")

    async def _process_single_business(
            self, business_data: Dict[str, Any]) -> BusinessOwnerInfo:
        """Process a single business to extract owner information using fully async processing."""
        business_name = business_data.get("business_name", "")
        state = business_data.get("state", "")
        zip_code = business_data.get("zip_code", "")

        try:
            # Setup progress tracking and logging
            logger.info(
                f"Processing business: {business_name} ({state} {zip_code})")

            # Initialize or retrieve processing memory
            memory = ProcessingMemory(
                business_name=business_name,
                state=state,
                zip_code=str(
                    zip_code),  # Convert to string to ensure proper typing
            )

            # 1. Generate search queries using async streaming
            logger.info(
                f"Step 1: Generating search queries for {business_name}")
            query_generation_start = time.time()
            queries = await generate_search_queries(self.llm, business_data)
            logger.info(
                f"Generated {len(queries)} search queries in {time.time() - query_generation_start:.2f}s"
            )

            # 2. Execute search queries in parallel with streaming results
            logger.info(
                f"Step 2: Executing search queries in parallel for {business_name}"
            )
            search_start = time.time()

            # Create an async queue for streaming search results as they complete
            search_queue = asyncio.Queue()

            # Create tasks that will put results in the queue when done
            async def process_query_and_enqueue(query_idx, query):
                try:
                    results = await search_with_query(query)
                    await search_queue.put((True, query_idx, results))
                except Exception as e:
                    logger.error(
                        f"Error in search query {query_idx}: {str(e)}")
                    await search_queue.put((False, query_idx, e))

            # Start all search tasks
            search_tasks = []
            for i, query in enumerate(queries):
                task = asyncio.create_task(process_query_and_enqueue(i, query))
                search_tasks.append(task)

            # Track when all search tasks are done
            # Don't wrap asyncio.gather in create_task - it already returns a future
            search_done = asyncio.gather(*search_tasks)

            # Process results as they become available
            all_search_results = []
            remaining = len(queries)

            # Stream the results as they complete
            while remaining > 0:
                success, idx, results = await search_queue.get()
                remaining -= 1

                if success:
                    # Access the query_type safely
                    if idx < len(queries):
                        query_type = queries[idx].query_type
                        search_results_list = (
                            results  # This is already a list of SearchResult objects
                        )
                        all_search_results.extend(search_results_list)
                        logger.info(
                            f"Got {len(search_results_list)} results for query type: {query_type}"
                        )
                    else:
                        # Fallback in case of index error
                        search_results_list = results
                        all_search_results.extend(search_results_list)
                        logger.info(
                            f"Got {len(search_results_list)} results for query at unknown index"
                        )
                else:
                    logger.error(
                        f"Search failed for query {idx}: {str(results)}")

            # Update memory with search results
            memory.search_results = all_search_results
            memory.processing_stage = "search_completed"
            logger.info(
                f"Completed all searches in {time.time() - search_start:.2f}s")

            # 3. Fetch content from top URLs with streaming results
            logger.info(
                f"Step 3: Fetching content from top URLs for {business_name}")
            fetch_start = time.time()

            top_urls = []
            # Get top results for each query type
            query_types = set(result.query_type
                              for result in all_search_results)
            for query_type in query_types:
                # Get results for this query type and sort by relevance
                type_results = [
                    r for r in all_search_results if r.query_type == query_type
                ]
                type_results.sort(key=lambda x: x.relevance_score,
                                  reverse=True)
                # Take top 3 results for each query type
                top_urls.extend([r.search_url for r in type_results[:3]])

            # Remove duplicates while preserving order
            top_urls = list(dict.fromkeys(top_urls))
            logger.info(f"Fetching content from {len(top_urls)} URLs")

            # Create a queue for streaming content results
            content_queue = asyncio.Queue()
            content_map = {}

            # Create tasks that put content in the queue when done
            async def fetch_url_and_enqueue(url_idx, url):
                try:
                    # Import PlaywrightURLLoader locally to handle import errors cleanly
                    from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader

                    logger.info(f"Fetching content from URL {url_idx}: {url}")

                    # Use LangChain's PlaywrightURLLoader for a single URL
                    loader = PlaywrightURLLoader(urls=[url],
                                                 remove_selectors=[
                                                     "nav", "header", "footer",
                                                     ".ads", "#ads",
                                                     ".navigation", ".menu"
                                                 ],
                                                 headless=True)

                    # Load document
                    try:
                        docs = loader.load()
                        if docs and len(docs) > 0:
                            content = docs[0].page_content
                            await content_queue.put((True, url, content))
                            logger.info(
                                f"Successfully loaded content from {url}")
                        else:
                            await content_queue.put(
                                (False, url, "No content retrieved"))
                            logger.warning(f"No content retrieved from {url}")
                    except Exception as load_error:
                        await content_queue.put(
                            (False, url, f"Load error: {str(load_error)}"))
                        logger.error(
                            f"Error loading content from {url}: {str(load_error)}"
                        )

                except Exception as e:
                    await content_queue.put((False, url, f"Error: {str(e)}"))
                    logger.error(f"General error processing {url}: {str(e)}")

            # Start all content fetch tasks
            content_tasks = []
            for i, url in enumerate(top_urls):
                task = asyncio.create_task(fetch_url_and_enqueue(i, url))
                content_tasks.append(task)

            # Track when all content fetch tasks are done
            # Don't wrap asyncio.gather in create_task - it already returns a future
            content_done = asyncio.gather(*content_tasks)

            # Process results as they become available
            remaining_urls = len(top_urls)

            # Stream the content as it's fetched
            while remaining_urls > 0:
                success, url, content = await content_queue.get()
                remaining_urls -= 1

                if success:
                    content_map[url] = content
                    logger.info(f"Fetched content from {url}")
                else:
                    content_map[url] = f"Failed to load content: {content}"
                    logger.error(f"Failed to fetch {url}: {content}")

            logger.info(
                f"Completed all content fetching in {time.time() - fetch_start:.2f}s"
            )

            # 4. Preprocess content with streaming
            logger.info(f"Step 4: Preprocessing content for {business_name}")
            preprocess_start = time.time()

            # Process chunks in a streaming fashion
            chunks = []

            # Create a queue for streaming preprocessed chunks
            chunk_queue = asyncio.Queue()

            async def preprocess_and_enqueue(url, content):
                try:
                    # Simple chunking by paragraphs
                    paragraphs = [
                        p for p in content.split("\n\n") if p.strip()
                    ]

                    for i, paragraph in enumerate(paragraphs):
                        # Only keep chunks with some minimum content
                        if len(paragraph.strip()) < 50:
                            continue

                        chunk = ContentChunk(
                            business_name=business_name,
                            source_url=url,
                            content=paragraph.strip(),
                            chunk_index=i,
                        )
                        await chunk_queue.put((True, chunk))
                except Exception as e:
                    logger.error(
                        f"Error preprocessing content from {url}: {str(e)}")
                    await chunk_queue.put((False, f"Error: {str(e)}"))

            # Start all preprocessing tasks
            preprocess_tasks = []
            total_items = sum(1 for content in content_map.values() if content)

            for url, content in content_map.items():
                if not content:
                    continue
                task = asyncio.create_task(preprocess_and_enqueue(
                    url, content))
                preprocess_tasks.append(task)

            # Track when all preprocessing tasks are done
            # Don't wrap asyncio.gather in create_task - it already returns a future
            preprocess_done = asyncio.gather(*preprocess_tasks)
            await preprocess_done

            # Get all chunks from the queue
            while not chunk_queue.empty():
                success, item = await chunk_queue.get()
                if success:
                    chunks.append(item)

            logger.info(
                f"Created {len(chunks)} content chunks in {time.time() - preprocess_start:.2f}s"
            )

            # Update memory with content chunks
            memory.content_chunks = chunks
            memory.processing_stage = "content_processed"

            # 5. Extract business information with async streaming
            logger.info(
                f"Step 5: Extracting business information for {business_name}")
            extraction_start = time.time()
            business_info = await extract_business_info(
                self.llm, business_data, memory)
            logger.info(
                f"Extracted business information in {time.time() - extraction_start:.2f}s"
            )

            logger.info(
                f"Extracted business info for {business_name} with confidence {business_info.confidence_score}"
            )

            # 6. Update memory with final state
            memory.processing_stage = "extraction_completed"

            return business_info

        except Exception as e:
            logger.error(
                f"Error processing business {business_name}: {str(e)}")
            # Return default info with error and complete error details in analysis notes
            return BusinessOwnerInfo(
                business_name=business_name,
                owner_name=f"Error: {str(e)}",
                primary_address="Processing failed",
                state=state,
                zip_code=str(zip_code),
                confidence_score=0.0,
                sources=[],
                analysis_notes=f"Processing error: {str(e)}\nTraceback: {traceback.format_exc()}",
                owner_confidence=0.0,
                address_confidence=0.0,
                alternative_owners=[],
            )

    async def process_businesses(
            self,
            df: pd.DataFrame,
            max_concurrency: Optional[int] = None) -> List[BusinessOwnerInfo]:
        """Process all businesses in parallel for maximum speed.

        The max_concurrency parameter is kept for API compatibility but is now optional.
        All businesses are processed in parallel without limiting concurrency.
        """
        results = []
        completed_count = 0
        total_count = len(df)

        # Create tasks for all businesses without using a semaphore
        tasks = []
        for _, row in df.iterrows():
            business_data = row.to_dict()
            tasks.append(self._process_single_business(business_data))

        # Process all tasks in parallel with maximum concurrency
        # Use gather instead of as_completed for maximum parallelism
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process the results
        for result in all_results:
            if isinstance(result, Exception):
                # Handle exception case
                logger.error(f"Error processing business: {str(result)}")
                # Add a placeholder result with full error details
                placeholder_result = BusinessOwnerInfo(
                    business_name="Error",
                    owner_name=f"Error: {str(result)}",
                    primary_address="Processing failed",
                    state="",
                    zip_code="",
                    confidence_score=0.0,
                    sources=[],
                    analysis_notes=f"Processing error: {str(result)}",
                    owner_confidence=0.0,
                    address_confidence=0.0,
                    alternative_owners=[],
                )
                results.append(placeholder_result)
            else:
                # Normal result
                results.append(result)

            completed_count += 1
            logger.info(
                f"Completed {completed_count}/{total_count} businesses")

        return results


async def _process_businesses_streaming(extractor: "BusinessInfoExtractor",
                                        df: pd.DataFrame):
    """Stream business processing results as they complete for maximum async performance.

    This generator processes businesses in parallel but yields results as soon as they're ready.
    """
    # Create a queue to hold results as they complete
    queue = asyncio.Queue()
    total = len(df)
    completed = 0

    # Define the processing function outside the loop to avoid closure issues
    async def process_and_enqueue(data):
        try:
            result = await extractor._process_single_business(data)
            await queue.put((True, result))
        except Exception as e:
            error_msg = (
                f"Error processing {data.get('business_name', 'unknown')}: {str(e)}"
            )
            logger.error(error_msg)
            console.print(f"[bold red]Error:[/bold red] {error_msg}")
            # Create a placeholder result for errors with complete analysis notes
            placeholder = BusinessOwnerInfo(
                business_name=data.get("business_name", "Error"),
                owner_name=f"Error: {str(e)}",
                primary_address="Processing failed",
                state=data.get("state", ""),
                zip_code=data.get("zip_code", ""),
                confidence_score=0.0,
                sources=[],
                analysis_notes=f"Processing error: {str(e)}\nTraceback: {traceback.format_exc()}",
                owner_confidence=0.0,
                address_confidence=0.0,
                alternative_owners=[],
            )
            await queue.put((False, placeholder))

    # Create tasks for all businesses
    tasks = []
    for _, row in df.iterrows():
        business_data = row.to_dict()

        # Add the task to our list
        task = asyncio.create_task(process_and_enqueue(business_data))
        tasks.append(task)

    # Create a done_evt that will be set when all tasks are complete
    done_evt = asyncio.Event()

    # Start a task to monitor for completion and set the event
    async def monitor_completion():
        try:
            # Gather any errors but continue processing all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Check for any exceptions that weren't already handled
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed: {str(result)}")
        except Exception as e:
            logger.error(f"Error in monitor_completion: {str(e)}")
        finally:
            # Always set the event to prevent hanging
            done_evt.set()

    monitor_task = asyncio.create_task(monitor_completion())

    # Yield results as they become available
    try:
        while completed < total:
            # Wait for either a result or completion
            get_next = asyncio.create_task(queue.get())
            wait_done = asyncio.create_task(done_evt.wait())

            done, pending = await asyncio.wait(
                [get_next, wait_done], return_when=asyncio.FIRST_COMPLETED)

            # Cancel the pending task
            for task in pending:
                task.cancel()

            # If we got a result, yield it
            if get_next in done:
                try:
                    queue_result = await get_next
                    # Unpack the tuple safely
                    if isinstance(queue_result,
                                  tuple) and len(queue_result) == 2:
                        success, result = queue_result
                        completed += 1

                        # Rich console logging with detailed status
                        if success:
                            logger.info(
                                f"Completed {completed}/{total}: {result.business_name}"
                            )
                        else:
                            logger.warning(
                                f"Failed {completed}/{total}: {result.business_name}"
                            )

                        yield result
                    else:
                        logger.error(
                            f"Unexpected queue result format: {queue_result}")
                except Exception as e:
                    logger.error(f"Error processing queue result: {str(e)}")
                    # Continue the loop to avoid hanging

            # If done_evt is set and no more results, we're finished
            if done_evt.is_set() and queue.empty() and completed >= total:
                break

    finally:
        # Clean up any remaining tasks
        for task in tasks:
            task.cancel()

        # Cancel the monitor task
        monitor_task.cancel()


async def run_extraction_workflow(
    excel_path: str,
    output_path: str = "business_owners.csv",
    max_concurrency: Optional[int] = None,
) -> str:
    """Run the complete extraction workflow from Excel file to CSV output with maximum parallelism.

    This function processes all businesses in parallel, with no rate limiting or concurrency constraints
    to achieve maximum speed. The max_concurrency parameter is kept for API compatibility but is ignored.

    Args:
        excel_path: Path to the Excel file containing business data
        output_path: Path to save the output CSV (default generates a timestamped filename)
        max_concurrency: Optional parameter kept for backward compatibility (ignored)

    Returns:
        Path to the output CSV file
    """
    start_time = time.time()
    logger.info(
        f"Starting business information extraction workflow from {excel_path} with maximum parallelism"
    )

    try:
        # Read Excel file in parallel-friendly way using pandas
        # Read Excel file using pandas with optimized options
        df = pd.read_excel(
            excel_path,
            engine=
            "openpyxl",  # Use openpyxl for better performance with xlsx files
            dtype={
                "Business Zip": str  # Pre-convert zip codes to strings
            },
        )
        logger.info(f"Loaded {len(df)} businesses from Excel file")

        # Map column names
        column_mapping = {
            "Business": "business_name",
            "Business ST": "state",
            "Business Zip": "zip_code",
        }

        # Rename columns to standard names (inplace for speed)
        df.rename(columns=column_mapping, inplace=True)

        # Check required columns
        required_columns = ["business_name", "state", "zip_code"]
        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]
        if missing_columns:
            logger.error(
                f"Missing required columns in Excel file: {missing_columns}")
            raise ValueError(
                f"Excel file missing required columns: {missing_columns}")

        # Initialize extractor with performance optimizations
        extractor = BusinessInfoExtractor()

        # Process all businesses in parallel with no restrictions using async streaming
        with console.status("[bold green]Processing businesses in parallel...",
                            spinner="dots") as status:
            console.print(
                Panel(
                    "[bold]Starting parallel business processing[/bold]",
                    title="[blue]Business Extraction Workflow[/blue]",
                    expand=False,
                ))

            # Create a progress bar for visual feedback
            with progress:
                task_id = progress.add_task("[cyan]Processing businesses...",
                                            total=len(df))

                # Use async streaming to process businesses - this provides maximum parallelism
                # with controlled streaming output
                console.print(
                    "[bold blue]Starting maximum parallelism business processing[/bold blue]"
                )
                results = []
                async for business_info in _process_businesses_streaming(
                        extractor, df):
                    results.append(business_info)
                    progress.update(task_id, advance=1)
                    # Rich console output with status information
                    if (business_info.owner_name
                            and "Error:" not in business_info.owner_name):
                        console.print(
                            f"[green]✓[/green] Processed: {business_info.business_name} - Owner: {business_info.owner_name}"
                        )
                    else:
                        console.print(
                            f"[yellow]⚠[/yellow] Processed with limited data: {business_info.business_name}"
                        )

        # Create output DataFrame directly (more efficient)
        console.print("[bold green]Creating output data...[/bold green]")
        output_data = []
        for info in results:
            output_data.append({
                "business_name": info.business_name,
                "owner_name": info.owner_name,
                "primary_address": info.primary_address,
                "state": info.state,
                "zip_code": info.zip_code,
                "confidence_score": info.confidence_score,
                "owner_confidence": info.owner_confidence,
                "address_confidence": info.address_confidence,
                "alternative_owners": "; ".join(info.alternative_owners) if info.alternative_owners else "",
                "sources": "; ".join(info.sources),
                "analysis_notes": info.analysis_notes or "",
            })

        output_df = pd.DataFrame(output_data)

        # Generate output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"business_owners_{timestamp}.csv"

        # Save to CSV with optimized settings
        output_df.to_csv(output_path, index=False)

        # Generate analysis summary for the React agent
        total_businesses = len(results)
        successful_count = sum(1 for info in results 
                              if info.owner_name and "Error:" not in info.owner_name 
                              and info.owner_confidence > 0.6)
        partial_count = sum(1 for info in results 
                           if info.owner_name and "Error:" not in info.owner_name 
                           and info.owner_confidence <= 0.6)
        failed_count = total_businesses - successful_count - partial_count
        
        # Calculate average confidence scores
        avg_confidence = sum(info.confidence_score for info in results) / total_businesses if total_businesses > 0 else 0
        avg_owner_confidence = sum(info.owner_confidence for info in results 
                                  if info.owner_name and "Error:" not in info.owner_name) / successful_count if successful_count > 0 else 0
        avg_address_confidence = sum(info.address_confidence for info in results 
                                     if info.primary_address and "Error:" not in info.primary_address) / successful_count if successful_count > 0 else 0
        
        # Create analysis summary for React agent
        analysis_summary = f"""
        EXTRACTION RESULTS SUMMARY:
        - Total businesses processed: {total_businesses}
        - Successfully extracted with high confidence: {successful_count} ({successful_count/total_businesses*100:.1f}%)
        - Partial extraction with lower confidence: {partial_count} ({partial_count/total_businesses*100:.1f}%)
        - Failed extractions: {failed_count} ({failed_count/total_businesses*100:.1f}%)
        
        CONFIDENCE METRICS:
        - Average overall confidence: {avg_confidence:.2f}
        - Average owner information confidence: {avg_owner_confidence:.2f}
        - Average address information confidence: {avg_address_confidence:.2f}
        
        PROCESSING PERFORMANCE:
        - Total processing time: {time.time() - start_time:.2f} seconds
        - Processing rate: {total_businesses/(time.time() - start_time):.2f} businesses/second
        """
        
        # Update the global status with the analysis summary
        if "_extraction_status" in globals():
            globals()["_extraction_status"]["analysis_summary"] = analysis_summary
        
        elapsed_time = time.time() - start_time
        logger.info(
            f"Extraction workflow completed in {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to {output_path}")
        logger.info(
            f"Processed {len(results)} businesses at {len(results)/elapsed_time:.2f} businesses/second"
        )

        return output_path

    except Exception as e:
        logger.error(f"Error in extraction workflow: {str(e)}")
        raise
