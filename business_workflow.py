"""
Business Information Extraction Workflow

This module implements an advanced async workflow to extract business owner information
using multi-query generation, content crawling, RAG and reflection for maximum accuracy.
The workflow uses LangMem to store and retrieve semantic information between steps.
"""
import os
import logging
import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_manager, create_memory_store_manager
from langmem.utils import NamespaceTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("business_workflow")

# Define Pydantic schemas for structured outputs
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

# Async utility functions for processing
async def generate_search_queries(
    llm: ChatOpenAI, 
    business_data: Dict[str, Any]
) -> List[BusinessQuery]:
    """Generate multiple search queries for different aspects of business information."""
    logger.info(f"Generating queries for {business_data['Business']}")
    
    template = """
    You are an expert at creating effective search queries to find business ownership information.
    
    For the following business, create 3 different search queries that will help find:
    1. The business owner's name
    2. The primary business address
    3. Business registration details
    
    Business: {business_name}
    State: {state}
    Zip Code: {zip_code}
    
    For each search query, generate a structured JSON output with the following format:
    
    {format_instructions}
    
    Be specific and include location details to increase the chance of finding the correct business.
    """
    
    parser = PydanticOutputParser(pydantic_object=List[BusinessQuery])
    prompt = PromptTemplate(
        template=template,
        input_variables=["business_name", "state", "zip_code"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    formatted_prompt = prompt.format(
        business_name=business_data['Business'],
        state=business_data['Business ST'],
        zip_code=business_data['Business Zip']
    )
    
    response = await llm.ainvoke(formatted_prompt)
    queries = parser.parse(response.content)
    
    return queries

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def search_with_query(query: BusinessQuery) -> List[SearchResult]:
    """Perform search using SerpAPI with retries and error handling."""
    logger.info(f"Searching for: {query.search_query}")
    
    search = SerpAPIWrapper()
    results = []
    
    try:
        raw_results = search.results(query.search_query)
        organic_results = raw_results.get("organic_results", [])
        
        for i, result in enumerate(organic_results[:5]):  # Process top 5 results
            results.append(
                SearchResult(
                    business_name=query.business_name,
                    query_type=query.query_type,
                    search_url=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    relevance_score=1.0 - (i * 0.15)  # Simple relevance score based on rank
                )
            )
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
    
    return results

async def fetch_content(urls: List[str]) -> Dict[str, str]:
    """Fetch and extract content from URLs using Playwright."""
    logger.info(f"Fetching content from {len(urls)} URLs")
    
    if not urls:
        return {}
    
    try:
        # Use PlaywrightURLLoader for JS-rendered content
        loader = PlaywrightURLLoader(
            urls=urls, 
            remove_selectors=["nav", "header", "footer", ".ad", "#cookie-banner"],
            continue_on_failure=True,
            headless=True
        )
        documents = await asyncio.to_thread(loader.load)
        
        # Create a map of URL to content
        content_map = {doc.metadata.get("source", ""): doc.page_content for doc in documents}
        return content_map
    
    except Exception as e:
        logger.error(f"Error fetching content: {str(e)}")
        return {}

async def preprocess_content(
    content_map: Dict[str, str],
    business_name: str
) -> List[ContentChunk]:
    """Split and preprocess content into chunks for embedding."""
    logger.info(f"Preprocessing content for {business_name}")
    chunks = []
    
    for url, content in content_map.items():
        # Simple chunking by paragraphs
        paragraphs = [p for p in content.split("\n\n") if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue
                
            chunk = ContentChunk(
                business_name=business_name,
                source_url=url,
                content=paragraph,
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
    logger.info(f"Scoring {len(chunks)} content chunks")
    
    if not chunks:
        return []
    
    # Get query embedding
    query_embedding = await asyncio.to_thread(
        lambda: embeddings.embed_query(query)
    )
    
    # Get embeddings for all chunks
    chunk_texts = [chunk.content for chunk in chunks]
    chunk_embeddings = await asyncio.to_thread(
        lambda: embeddings.embed_documents(chunk_texts)
    )
    
    # Calculate similarity scores
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_embedding = chunk_embeddings[i]
        
        # Cosine similarity
        similarity = sum(a * b for a, b in zip(query_embedding, chunk_embedding)) / (
            (sum(a * a for a in query_embedding) ** 0.5) * 
            (sum(b * b for b in chunk_embedding) ** 0.5)
        )
        
        scored_chunks.append((chunk, similarity))
    
    # Sort by score in descending order
    return sorted(scored_chunks, key=lambda x: x[1], reverse=True)

async def extract_business_info(
    llm: ChatOpenAI,
    business_data: Dict[str, Any],
    memory: ProcessingMemory
) -> BusinessOwnerInfo:
    """Extract structured business owner information from processed data."""
    logger.info(f"Extracting business info for {business_data['Business']}")
    
    # Prepare context from memory
    context = []
    
    # Add search results
    for result in memory.search_results:
        context.append(f"Search Result ({result.query_type}): {result.snippet}")
    
    # Add content chunks (sorted by relevance)
    content_context = []
    for i, chunk in enumerate(memory.content_chunks[:10]):  # Use top 10 chunks
        content_context.append(f"Content {i+1} from {chunk.source_url}: {chunk.content}")
    
    # Add candidates if available
    owner_candidates = []
    for name, score in memory.owner_name_candidates:
        owner_candidates.append(f"Owner candidate: {name} (confidence: {score:.2f})")
    
    address_candidates = []
    for addr, score in memory.address_candidates:
        address_candidates.append(f"Address candidate: {addr} (confidence: {score:.2f})")
    
    # Build the prompt
    template = """
    You are an expert at extracting business owner information from various sources.
    
    Based on the following information about a business, extract the business owner's name
    and primary business address. If the information is not available, indicate this clearly.
    
    Business: {business_name}
    State: {state}
    Zip Code: {zip_code}
    
    Search Results:
    {search_results}
    
    Content Excerpts:
    {content_excerpts}
    
    Owner Name Candidates:
    {owner_candidates}
    
    Address Candidates:
    {address_candidates}
    
    Provide your answer in the following JSON format:
    {format_instructions}
    
    Ensure high accuracy and only include information you're confident about.
    Set the confidence score based on how certain you are about the information extracted.
    """
    
    parser = PydanticOutputParser(pydantic_object=BusinessOwnerInfo)
    
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "business_name", "state", "zip_code", "search_results", 
            "content_excerpts", "owner_candidates", "address_candidates"
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    formatted_prompt = prompt.format(
        business_name=business_data['Business'],
        state=business_data['Business ST'],
        zip_code=business_data['Business Zip'],
        search_results="\n".join(context),
        content_excerpts="\n".join(content_context),
        owner_candidates="\n".join(owner_candidates),
        address_candidates="\n".join(address_candidates)
    )
    
    response = await llm.ainvoke(formatted_prompt)
    business_info = parser.parse(response.content)
    
    # Set additional fields
    if not business_info.state:
        business_info.state = business_data['Business ST']
    if not business_info.zip_code:
        business_info.zip_code = str(business_data['Business Zip'])
        
    # Add sources
    business_info.sources = list({result.search_url for result in memory.search_results})
    
    return business_info

class BusinessInfoExtractor:
    """Main class to orchestrate the business information extraction workflow."""
    
    def __init__(self):
        """Initialize the business information extractor."""
        # Check for API keys
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required in environment variables")
        if not os.environ.get("SERPAPI_API_KEY"):
            raise ValueError("SERPAPI_API_KEY is required in environment variables")
            
        # Initialize language model
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            streaming=True
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        # Memory store setup
        self.store = InMemoryStore(
            index={
                "dims": 1536,  # Dimension for text-embedding-3-small
                "embed": "openai:text-embedding-3-small"  # Specify the embedding model
            }
        )
        
        # Set up namespaces for different memory types
        self.query_ns = NamespaceTemplate(("queries", "{business_id}"))
        self.processing_ns = NamespaceTemplate(("processing", "{business_id}"))
        
        # Memory managers for extracting and storing query and processing information
        self.query_manager = create_memory_store_manager(
            "gpt-4-turbo",
            schemas=[QueryGenerator],
            instructions="Track and generate search queries for business information extraction.",
            namespace=self.query_ns,
            store=self.store,
            enable_inserts=True,
            enable_deletes=True
        )
        
        self.processing_manager = create_memory_store_manager(
            "gpt-4-turbo",
            schemas=[ProcessingMemory],
            instructions="Track processing state and extracted information for each business.",
            namespace=self.processing_ns,
            store=self.store,
            enable_inserts=True,
            enable_deletes=True
        )
        
        logger.info("Business information extractor initialized")
    
    async def _process_single_business(self, business_data: Dict[str, Any]) -> BusinessOwnerInfo:
        """Process a single business to extract owner information."""
        business_id = f"{business_data['Business'].lower().replace(' ', '_')}_{business_data['Business Zip']}"
        business_name = business_data['Business']
        
        logger.info(f"Processing business: {business_name}")
        
        # Initialize processing memory
        processing_memory = ProcessingMemory(
            business_name=business_name,
            state=business_data['Business ST'],
            zip_code=str(business_data['Business Zip']),
            processing_stage="query_generation"
        )
        
        # Step 1: Generate search queries
        queries = await generate_search_queries(self.llm, business_data)
        
        # Store queries in memory
        query_memory = QueryGenerator(
            business_name=business_name,
            queries=queries
        )
        
        # Namespace config for storing in memory
        config = {"configurable": {"business_id": business_id}}
        await self.query_manager.acreate([query_memory], config=config)
        
        # Step 2: Execute searches in parallel
        search_tasks = [search_with_query(query) for query in queries]
        search_results_lists = await asyncio.gather(*search_tasks)
        
        # Flatten search results
        all_search_results = []
        for results in search_results_lists:
            all_search_results.extend(results)
            
        # Update processing memory
        processing_memory.search_results = all_search_results
        processing_memory.processing_stage = "content_fetching"
        
        # Step 3: Fetch content from search results
        urls = list({result.search_url for result in all_search_results})
        content_map = await fetch_content(urls)
        
        # Step 4: Preprocess content
        content_chunks = await preprocess_content(content_map, business_name)
        
        # Step 5: Score content against business query
        query_text = f"{business_name} owner information {business_data['Business ST']} {business_data['Business Zip']}"
        scored_chunks = await score_and_rank_chunks(content_chunks, query_text, self.embeddings)
        
        # Update processing memory with ranked chunks
        processing_memory.content_chunks = [chunk for chunk, _ in scored_chunks]
        processing_memory.processing_stage = "information_extraction"
        
        # Extract owner name candidates from content
        owner_prompt = PromptTemplate(
            template="Extract possible business owner names from this text: {text}",
            input_variables=["text"]
        )
        
        # Extract address candidates from content
        address_prompt = PromptTemplate(
            template="Extract possible business addresses from this text: {text}",
            input_variables=["text"]
        )
        
        # Get top chunks for candidate extraction
        top_chunks = [chunk for chunk, score in scored_chunks[:5]]
        
        # Extract candidates in parallel (if there are chunks)
        if top_chunks:
            owner_tasks = []
            address_tasks = []
            
            for chunk in top_chunks:
                owner_tasks.append(self.llm.ainvoke(owner_prompt.format(text=chunk.content)))
                address_tasks.append(self.llm.ainvoke(address_prompt.format(text=chunk.content)))
                
            owner_responses = await asyncio.gather(*owner_tasks)
            address_responses = await asyncio.gather(*address_tasks)
            
            # Process candidate names
            for i, response in enumerate(owner_responses):
                owner_name = response.content.strip()
                if owner_name and owner_name.lower() not in ["none", "not found", "unknown"]:
                    # Add with score based on chunk ranking
                    score = 1.0 - (i * 0.15)
                    processing_memory.owner_name_candidates.append((owner_name, score))
            
            # Process candidate addresses
            for i, response in enumerate(address_responses):
                address = response.content.strip()
                if address and address.lower() not in ["none", "not found", "unknown"]:
                    # Add with score based on chunk ranking
                    score = 1.0 - (i * 0.15)
                    processing_memory.address_candidates.append((address, score))
        
        # Store processing state in memory
        await self.processing_manager.acreate([processing_memory], config=config)
        
        # Step 6: Extract final business information
        business_info = await extract_business_info(self.llm, business_data, processing_memory)
        
        # Step 7: Update processing memory with final state
        processing_memory.processing_stage = "completed"
        await self.processing_manager.aupdate([processing_memory], config=config)
        
        return business_info
    
    async def process_businesses(self, df: pd.DataFrame, max_concurrency: int = 5) -> List[BusinessOwnerInfo]:
        """Process a batch of businesses with controlled concurrency."""
        logger.info(f"Processing {len(df)} businesses with max concurrency {max_concurrency}")
        
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_with_semaphore(business_data):
            async with semaphore:
                return await self._process_single_business(business_data)
        
        # Create tasks for all businesses
        tasks = []
        for _, row in df.iterrows():
            business_data = row.to_dict()
            tasks.append(process_with_semaphore(business_data))
        
        # Execute tasks with controlled concurrency
        results = await asyncio.gather(*tasks)
        return results

async def run_extraction_workflow(excel_path: str, output_path: str = "business_owners.csv"):
    """Run the complete extraction workflow from Excel file to CSV output."""
    logger.info(f"Starting extraction workflow from {excel_path}")
    
    # Load business data
    df = pd.read_excel(excel_path)
    logger.info(f"Loaded {len(df)} businesses from Excel")
    
    # Initialize extractor
    extractor = BusinessInfoExtractor()
    
    # Process businesses
    results = await extractor.process_businesses(df)
    logger.info(f"Processed {len(results)} businesses")
    
    # Convert results to DataFrame
    output_data = []
    for info in results:
        output_data.append({
            "Business Name": info.business_name,
            "Owner Name": info.owner_name or "Not found",
            "Primary Address": info.primary_address or "Not found",
            "State": info.state,
            "Zip Code": info.zip_code,
            "Confidence Score": info.confidence_score,
            "Sources": "; ".join(info.sources)
        })
    
    output_df = pd.DataFrame(output_data)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    return output_path

# Entry point when running as a script
if __name__ == "__main__":
    asyncio.run(run_extraction_workflow("sample.xlsx"))