"""
Business Information Extraction Workflow

This module implements an advanced async workflow to extract business owner information
using multi-query generation, content crawling, RAG and reflection for maximum accuracy.
The workflow uses LangMem to store and retrieve semantic information between steps.
"""
import os
import logging
import asyncio
import time
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

from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END
# Custom in-memory solution since langgraph.store.memory.MemorySolutionStore is not available
class SimpleMemoryStore:
    """A simple in-memory key-value store for business extraction workflow."""
    def __init__(self):
        self._store = {}
    
    def get(self, key, default=None):
        """Get a value from the store."""
        return self._store.get(key, default)
    
    def set(self, key, value):
        """Set a value in the store."""
        self._store[key] = value
        return value
    
    def delete(self, key):
        """Delete a value from the store."""
        if key in self._store:
            del self._store[key]
    
    def list(self):
        """List all keys in the store."""
        return list(self._store.keys())

# Simplified function to get langmem store - returns our simple memory store
def get_langmem_store():
    """Get a memory store instance."""
    raise ValueError("Original langmem store not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessQuery(BaseModel):
    """A structured query format for SerpAPI searches."""
    business_name: str = Field(..., description="The full business name")
    location: str = Field(..., description="Business location including state and zip code")
    query_type: str = Field(..., description="Type of query: 'owner', 'address', 'contact'")
    search_query: str = Field(..., description="The full search query to submit to search engine")

    class Config:
        schema_extra = {
            "example": {
                "business_name": "Absolute Home Improvement LLC",
                "location": "MO 64040",
                "query_type": "owner",
                "search_query": "who owns Absolute Home Improvement LLC in Missouri 64040 owner name"
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
                "snippet": "Absolute Home Improvement LLC is owned by John Smith who started the company in 2015.",
                "relevance_score": 0.85
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
                "content": "John Smith is the founder and owner of Absolute Home Improvement LLC.",
                "chunk_index": 1,
                "embedding_id": "emb_12345"
            }
        }

class BusinessOwnerInfo(BaseModel):
    """Final extracted business owner information."""
    business_name: str
    owner_name: Optional[str] = Field(None, description="Full name of the business owner")
    primary_address: Optional[str] = Field(None, description="Primary business address")
    state: str
    zip_code: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)

    class Config:
        schema_extra = {
            "example": {
                "business_name": "Absolute Home Improvement LLC",
                "owner_name": "John Smith",
                "primary_address": "123 Main Street, Kansas City",
                "state": "MO",
                "zip_code": "64040",
                "confidence_score": 0.92,
                "sources": ["https://example.com/business/info", "https://example.com/directory"]
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
    owner_name_candidates: List[Tuple[str, float]] = Field(default_factory=list)
    address_candidates: List[Tuple[str, float]] = Field(default_factory=list)
    reflection_notes: Optional[str] = None

async def generate_search_queries(
    llm: ChatOpenAI, 
    business_data: Dict[str, Any]
) -> List[BusinessQuery]:
    """Generate multiple search queries for different aspects of business information."""
    # Extract business information
    business_name = business_data.get("business_name", "")
    state = business_data.get("state", "")
    zip_code = business_data.get("zip_code", "")
    location = f"{state} {zip_code}"
    
    # Create the prompt
    query_generation_prompt = ChatPromptTemplate.from_template("""
    You are a business research assistant specializing in finding detailed information about businesses.
    
    I need to research the following business:
    Business Name: {business_name}
    Location: {location}
    
    Generate 3 distinct search queries to find the following information:
    1. The owner or owners of the business (full names)
    2. The primary address of the business
    3. General business contact information
    
    Format your response as a JSON array of query objects with these fields:
    - business_name: The full business name
    - location: Business location information
    - query_type: Either 'owner', 'address', or 'contact'
    - search_query: The full search query to submit to search engine
    
    Make each query specific and targeted to find the exact information needed.
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
                search_query=query_data.get("search_query", "")
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
                search_query=f"who owns {business_name} in {location} owner name"
            ),
            BusinessQuery(
                business_name=business_name,
                location=location,
                query_type="address",
                search_query=f"{business_name} address {location} headquarters location"
            )
        ]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def search_with_query(query: BusinessQuery) -> List[SearchResult]:
    """Perform search using SerpAPI with retries and error handling."""
    try:
        # Initialize SerpAPI wrapper
        serpapi_key = os.environ.get("SERPAPI_API_KEY")
        if not serpapi_key:
            logger.error("SERPAPI_API_KEY not set")
            return []
        
        search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
        
        # Perform search
        search_result = search.results(query.search_query)
        
        # Process organic results
        results = []
        if "organic_results" in search_result:
            for i, result in enumerate(search_result["organic_results"][:5]):  # Limit to top 5 results
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")
                
                # Calculate a simple relevance score based on position
                position_score = 1.0 - (i * 0.15)  # 1.0, 0.85, 0.7, 0.55, 0.4
                
                # Add basic keywords matching for the relevance score
                business_keywords = query.business_name.lower().split()
                query_type_keywords = {"owner": ["owner", "founder", "ceo", "president"], 
                                    "address": ["address", "location", "headquarters", "office"],
                                    "contact": ["contact", "phone", "email", "website"]}
                
                keyword_matches = 0
                search_text = (title + " " + snippet).lower()
                
                # Check business name keywords
                for keyword in business_keywords:
                    if keyword in search_text:
                        keyword_matches += 1
                
                # Check query type specific keywords
                for keyword in query_type_keywords.get(query.query_type, []):
                    if keyword in search_text:
                        keyword_matches += 2  # Give more weight to query type matches
                
                # Calculate final relevance score
                keyword_score = min(0.5, keyword_matches * 0.1)  # Max 0.5 for keyword matches
                relevance_score = min(1.0, position_score + keyword_score)
                
                # Create SearchResult object
                search_result = SearchResult(
                    business_name=query.business_name,
                    query_type=query.query_type,
                    search_url=link,
                    snippet=snippet,
                    relevance_score=relevance_score
                )
                results.append(search_result)
        
        return results
    
    except Exception as e:
        logger.error(f"Error during search for {query.search_query}: {str(e)}")
        # Raise for retry
        raise

async def fetch_content(urls: List[str]) -> Dict[str, str]:
    """Fetch and extract content from URLs using Playwright's async API."""
    content_map = {}
    
    # Create batches to avoid overloading
    batch_size = 3
    url_batches = [urls[i:i+batch_size] for i in range(0, len(urls), batch_size)]
    
    try:
        # Import here to catch ImportError early
        from playwright.async_api import async_playwright
        from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader
    except ImportError:
        logger.error("Playwright not found. Cannot fetch content.")
        # Return empty content for all URLs
        return {url: f"Failed to load content for {url}" for url in urls}
    
    for batch in url_batches:
        try:
            # Use async Playwright directly to avoid sync issues
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                
                for url in batch:
                    try:
                        page = await context.new_page()
                        logger.info(f"Navigating to {url}")
                        response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        
                        if response and response.ok:
                            # Extract all text from the page
                            content = await page.evaluate("""() => {
                                // Remove unwanted elements
                                const elementsToRemove = document.querySelectorAll('nav, header, footer, .ads, #ads, .navigation, .menu');
                                for (let el of elementsToRemove) {
                                    if (el && el.parentNode) el.parentNode.removeChild(el);
                                }
                                
                                // Get the cleaned content
                                return document.body.innerText;
                            }""")
                            
                            content_map[url] = content
                        else:
                            status = response.status if response else "Unknown"
                            logger.warning(f"Failed to load {url} - Status: {status}")
                            content_map[url] = f"Failed to load content for {url} (Status: {status})"
                            
                        await page.close()
                    except Exception as page_error:
                        logger.error(f"Error fetching page {url}: {str(page_error)}")
                        content_map[url] = f"Error loading content: {str(page_error)}"
                
                await browser.close()
            
            # Wait between batches to avoid overloading
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error fetching content batch: {str(e)}")
            # Add error message content for all URLs in the failed batch
            for url in batch:
                if url not in content_map:
                    content_map[url] = f"Failed to load content: {str(e)}"
    
    # Ensure all URLs have an entry
    for url in urls:
        if url not in content_map:
            content_map[url] = "No content retrieved"
    
    logger.info(f"Fetched content from {len(content_map)} URLs")
    
    return content_map

async def preprocess_content(
    content_map: Dict[str, str],
    business_name: str
) -> List[ContentChunk]:
    """Split and preprocess content into chunks for embedding."""
    chunks = []
    
    for url, content in content_map.items():
        if not content:
            continue
        
        # Simple chunking by paragraphs
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            # Only keep chunks with some minimum content
            if len(paragraph.strip()) < 50:
                continue
                
            chunk = ContentChunk(
                business_name=business_name,
                source_url=url,
                content=paragraph.strip(),
                chunk_index=i
            )
            chunks.append(chunk)
    
    return chunks

async def score_and_rank_chunks(
    chunks: List[ContentChunk],
    query: str,
    embeddings: Embeddings
) -> List[Tuple[ContentChunk, float]]:
    """Score and rank content chunks based on relevance to query."""
    if not chunks:
        return []
    
    try:
        # Create documents from chunks
        docs = [Document(page_content=chunk.content, metadata={"index": i}) 
                for i, chunk in enumerate(chunks)]
        
        # Get query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Get document embeddings
        doc_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])
        
        # Calculate similarity scores
        scored_chunks = []
        for i, doc_embedding in enumerate(doc_embeddings):
            # Calculate cosine similarity
            similarity = sum(a*b for a, b in zip(query_embedding, doc_embedding)) / (
                sum(a*a for a in query_embedding)**0.5 * sum(b*b for b in doc_embedding)**0.5
            )
            
            chunk = chunks[i]
            scored_chunks.append((chunk, similarity))
        
        # Sort by score in descending order
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks
    
    except Exception as e:
        logger.error(f"Error scoring chunks: {str(e)}")
        # Return unsorted chunks with neutral scores as fallback
        return [(chunk, 0.5) for chunk in chunks]

async def extract_business_info(
    llm: ChatOpenAI,
    business_data: Dict[str, Any],
    memory: ProcessingMemory
) -> BusinessOwnerInfo:
    """Extract structured business owner information from processed data."""
    # Format all the content chunks and search results
    chunk_texts = []
    for chunk in memory.content_chunks:
        chunk_texts.append(f"SOURCE: {chunk.source_url}\nCONTENT: {chunk.content}")
    
    search_texts = []
    for result in memory.search_results:
        search_texts.append(f"QUERY TYPE: {result.query_type}\nURL: {result.search_url}\nSNIPPET: {result.snippet}")
    
    # Create the prompt
    extraction_prompt = ChatPromptTemplate.from_template("""
    You are a business information extraction specialist.
    
    BUSINESS INFORMATION:
    Business Name: {business_name}
    State: {state}
    Zip Code: {zip_code}
    
    SEARCH RESULTS:
    {search_results}
    
    CONTENT CHUNKS:
    {content_chunks}
    
    TASK:
    Based on ONLY the information provided above, extract the following details about the business:
    
    1. Business Owner's Full Name
    2. Primary Business Address
    
    If specific information is not clearly stated in the provided content, indicate with "Not found in sources".
    Calculate a confidence score from 0.0 to 1.0 for each extracted piece of information.
    
    Respond in JSON format:
    {{
        "business_name": "{business_name}",
        "owner_name": "Extracted owner name or 'Not found in sources'",
        "primary_address": "Extracted address or 'Not found in sources'",
        "state": "{state}",
        "zip_code": "{zip_code}",
        "confidence_score": 0.95,
        "sources": ["list", "of", "source", "urls", "that", "provided", "the", "information"]
    }}
    """)
    
    # Parse output as JSON
    output_parser = JsonOutputParser()
    
    # Create the chain
    extraction_chain = extraction_prompt | llm | output_parser
    
    # Extract information
    try:
        result = await extraction_chain.ainvoke({
            "business_name": business_data.get("business_name", ""),
            "state": business_data.get("state", ""),
            "zip_code": business_data.get("zip_code", ""),
            "search_results": "\n\n".join(search_texts),
            "content_chunks": "\n\n".join(chunk_texts)
        })
        
        # Convert to BusinessOwnerInfo object
        business_info = BusinessOwnerInfo(
            business_name=result.get("business_name", business_data.get("business_name", "")),
            owner_name=result.get("owner_name", "Not found in sources"),
            primary_address=result.get("primary_address", "Not found in sources"),
            state=result.get("state", business_data.get("state", "")),
            zip_code=result.get("zip_code", business_data.get("zip_code", "")),
            confidence_score=float(result.get("confidence_score", 0.0)),
            sources=result.get("sources", [])
        )
        
        return business_info
    
    except Exception as e:
        logger.error(f"Error extracting business info: {str(e)}")
        # Return default object with error indication
        return BusinessOwnerInfo(
            business_name=business_data.get("business_name", ""),
            owner_name="Error during extraction",
            primary_address="Error during extraction",
            state=business_data.get("state", ""),
            zip_code=business_data.get("zip_code", ""),
            confidence_score=0.0,
            sources=[]
        )

class BusinessInfoExtractor:
    """Main class to orchestrate the business information extraction workflow."""
    
    def __init__(self):
        """Initialize the business information extractor."""
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-1106",
            temperature=0
        )
        self.embeddings = OpenAIEmbeddings()
        # Initialize memory store
        try:
            self.store = get_langmem_store()
            logger.info("Initialized memory store for business extraction")
        except Exception as e:
            self.store = SimpleMemoryStore()
            logger.warning(f"Failed to initialize LangMem store, using simple in-memory store: {str(e)}")
    
    async def _process_single_business(self, business_data: Dict[str, Any]) -> BusinessOwnerInfo:
        """Process a single business to extract owner information."""
        business_name = business_data.get("business_name", "")
        state = business_data.get("state", "")
        zip_code = business_data.get("zip_code", "")
        
        logger.info(f"Processing business: {business_name} ({state} {zip_code})")
        
        # Initialize or retrieve processing memory
        memory = ProcessingMemory(
            business_name=business_name,
            state=state,
            zip_code=str(zip_code)  # Convert to string to ensure proper typing
        )
        
        try:
            # 1. Generate search queries
            queries = await generate_search_queries(self.llm, business_data)
            logger.info(f"Generated {len(queries)} search queries for {business_name}")
            
            # 2. Execute search queries
            all_search_results = []
            for query in queries:
                results = await search_with_query(query)
                all_search_results.extend(results)
                logger.info(f"Got {len(results)} search results for query type: {query.query_type}")
            
            # Update memory with search results
            memory.search_results = all_search_results
            memory.processing_stage = "search_completed"
            
            # 3. Fetch content from top URLs
            top_urls = []
            # Get top results for each query type
            query_types = set(result.query_type for result in all_search_results)
            for query_type in query_types:
                # Get results for this query type and sort by relevance
                type_results = [r for r in all_search_results if r.query_type == query_type]
                type_results.sort(key=lambda x: x.relevance_score, reverse=True)
                # Take top 3 results for each query type
                top_urls.extend([r.search_url for r in type_results[:3]])
            
            # Remove duplicates while preserving order
            top_urls = list(dict.fromkeys(top_urls))
            logger.info(f"Fetching content from {len(top_urls)} URLs for {business_name}")
            
            content_map = await fetch_content(top_urls)
            logger.info(f"Fetched content from {len(content_map)} URLs")
            
            # 4. Preprocess content
            chunks = await preprocess_content(content_map, business_name)
            logger.info(f"Created {len(chunks)} content chunks")
            
            # Update memory with content chunks
            memory.content_chunks = chunks
            memory.processing_stage = "content_processed"
            
            # 5. Extract business information
            business_info = await extract_business_info(self.llm, business_data, memory)
            logger.info(f"Extracted business info for {business_name} with confidence {business_info.confidence_score}")
            
            # 6. Update memory with final state
            memory.processing_stage = "extraction_completed"
            
            return business_info
            
        except Exception as e:
            logger.error(f"Error processing business {business_name}: {str(e)}")
            # Return default info with error
            return BusinessOwnerInfo(
                business_name=business_name,
                owner_name=f"Error: {str(e)}",
                primary_address="Processing failed",
                state=state,
                zip_code=str(zip_code),
                confidence_score=0.0,
                sources=[]
            )
    
    async def process_businesses(self, df: pd.DataFrame, max_concurrency: int = 5) -> List[BusinessOwnerInfo]:
        """Process a batch of businesses with controlled concurrency."""
        results = []
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_with_semaphore(business_data):
            async with semaphore:
                return await self._process_single_business(business_data)
        
        # Create tasks for all businesses
        tasks = []
        for _, row in df.iterrows():
            business_data = row.to_dict()
            tasks.append(process_with_semaphore(business_data))
        
        # Process all tasks and collect results
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            logger.info(f"Completed {len(results)}/{len(tasks)} businesses")
        
        return results

async def run_extraction_workflow(excel_path: str, output_path: str = "business_owners.csv", max_concurrency: int = 5) -> str:
    """Run the complete extraction workflow from Excel file to CSV output."""
    start_time = time.time()
    logger.info(f"Starting business information extraction workflow from {excel_path}")
    
    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        logger.info(f"Loaded {len(df)} businesses from Excel file")
        
        # Map column names
        column_mapping = {
            "Business": "business_name",
            "Business ST": "state",
            "Business Zip": "zip_code"
        }
        
        # Rename columns to standard names
        df = df.rename(columns=column_mapping)
        
        # Check required columns
        required_columns = ["business_name", "state", "zip_code"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in Excel file: {missing_columns}")
            raise ValueError(f"Excel file missing required columns: {missing_columns}")
        
        # Convert zip_code column to string to prevent type errors
        df["zip_code"] = df["zip_code"].astype(str)
        
        # Initialize extractor
        extractor = BusinessInfoExtractor()
        
        # Process businesses
        results = await extractor.process_businesses(df, max_concurrency)
        
        # Create output DataFrame
        output_df = pd.DataFrame([
            {
                "business_name": info.business_name,
                "owner_name": info.owner_name,
                "primary_address": info.primary_address,
                "state": info.state,
                "zip_code": info.zip_code,
                "confidence_score": info.confidence_score,
                "sources": "; ".join(info.sources)
            }
            for info in results
        ])
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_path:
            output_path = f"business_owners_{timestamp}.csv"
        output_df.to_csv(output_path, index=False)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Extraction workflow completed in {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error in extraction workflow: {str(e)}")
        raise