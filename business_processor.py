"""
Business Processor Integration Module

This module provides an interface between the business extraction workflow
and the React agent, allowing the agent to orchestrate the extraction process.
"""
import os
import logging
import asyncio
import pandas as pd
from typing import Dict, Any, List, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from business_workflow import BusinessOwnerInfo, run_extraction_workflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("business_processor")

# Global state to track processing
_processing_task = None
_processing_results = []
_processing_status = "idle"
_last_error = None

class BusinessExtractionInput(BaseModel):
    """Input for the business extraction process."""
    excel_path: str = Field(..., description="Path to the Excel file containing business data")
    max_concurrency: int = Field(5, description="Maximum number of businesses to process concurrently")

class BusinessExtractionResults(BaseModel):
    """Results of the business extraction process."""
    processed_count: int
    total_count: int
    status: str
    error: Optional[str] = None
    sample_results: List[Dict[str, Any]] = []
    output_path: Optional[str] = None

async def _run_extraction_process(excel_path: str, max_concurrency: int = 5) -> str:
    """Run the full extraction process and return the output file path."""
    global _processing_status, _processing_results, _last_error
    
    try:
        _processing_status = "processing"
        _processing_results = []
        
        # Generate a timestamped output path
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"business_owners_{timestamp}.csv"
        
        # Run the extraction workflow
        result_path = await run_extraction_workflow(excel_path, output_path)
        
        # Load results for status reporting
        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            _processing_results = df.to_dict('records')
            _processing_status = "completed"
            return result_path
        else:
            _processing_status = "error"
            _last_error = "Output file was not created"
            return ""
            
    except Exception as e:
        logger.error(f"Extraction process error: {str(e)}")
        _processing_status = "error"
        _last_error = str(e)
        return ""

def start_extraction_process(excel_path: str, max_concurrency: int = 5) -> BusinessExtractionResults:
    """Start the extraction process as a background task."""
    global _processing_task, _processing_status, _last_error
    
    # Cancel any existing task
    if _processing_task and not _processing_task.done():
        logger.warning("Cancelling existing extraction task")
        _processing_task.cancel()
    
    # Create a new task
    _processing_status = "starting"
    _last_error = None
    
    loop = asyncio.get_event_loop()
    _processing_task = loop.create_task(_run_extraction_process(excel_path, max_concurrency))
    
    # Return initial status
    return BusinessExtractionResults(
        processed_count=0,
        total_count=0,
        status=_processing_status,
        sample_results=[]
    )

def get_extraction_status() -> BusinessExtractionResults:
    """Get the current status of the extraction process."""
    global _processing_status, _processing_results, _last_error
    
    # Check if the task is complete
    if _processing_task and _processing_task.done():
        try:
            output_path = _processing_task.result()
            if output_path:
                _processing_status = "completed"
            else:
                _processing_status = "error"
        except asyncio.CancelledError:
            _processing_status = "cancelled"
        except Exception as e:
            _processing_status = "error"
            _last_error = str(e)
    
    # Count results
    processed_count = len(_processing_results)
    
    # Determine total count
    total_count = 0
    if _processing_status == "completed":
        total_count = processed_count
    elif processed_count > 0:
        # Estimate based on progress
        total_count = processed_count
    
    # Get a sample of results
    sample_results = _processing_results[:5] if _processing_results else []
    
    # Get output path if available
    output_path = None
    if _processing_task and _processing_task.done() and not _processing_task.exception():
        try:
            output_path = _processing_task.result()
        except:
            pass
    
    return BusinessExtractionResults(
        processed_count=processed_count,
        total_count=total_count,
        status=_processing_status,
        error=_last_error,
        sample_results=sample_results,
        output_path=output_path
    )

# Define LangChain tools for the agent to use
class ExtractBusinessInfoTool(BaseTool):
    """Tool for extracting business owner information from Excel file."""
    name: str = "extract_business_info"
    description: str = """
    Extracts business owner names and primary addresses for businesses in an Excel file.
    Uses advanced multi-query generation, web search, content crawling and RAG techniques.
    All processing is done asynchronously for efficient handling of large datasets.
    """
    args_schema = BusinessExtractionInput
    
    def _run(self, excel_path: str, max_concurrency: int = 5) -> Dict[str, Any]:
        """Run the business information extraction process."""
        # Start the extraction process
        result = start_extraction_process(excel_path, max_concurrency)
        return result.dict()
    
    async def _arun(self, excel_path: str, max_concurrency: int = 5) -> Dict[str, Any]:
        """Run the business information extraction process asynchronously."""
        # Use the synchronous version, which already manages async background tasks
        return self._run(excel_path, max_concurrency)

class CheckExtractionStatusTool(BaseTool):
    """Tool for checking the status of the business information extraction process."""
    name: str = "check_extraction_status"
    description: str = """
    Checks the current status of the business information extraction process.
    Returns progress information and sample results if available.
    """
    
    def _run(self) -> Dict[str, Any]:
        """Get the current status of the extraction process."""
        result = get_extraction_status()
        return result.dict()
    
    async def _arun(self) -> Dict[str, Any]:
        """Get the current status of the extraction process asynchronously."""
        return self._run()

# Get business extraction tools
def get_business_extraction_tools() -> List[BaseTool]:
    """Get tools for business information extraction."""
    return [
        ExtractBusinessInfoTool(),
        CheckExtractionStatusTool()
    ]