"""
Business Processor Integration Module

This module provides an interface between the business extraction workflow
and the React agent, allowing the agent to orchestrate the extraction process.
"""
import os
import asyncio
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Type
import pandas as pd
from datetime import datetime

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for extraction process
_extraction_task = None
_extraction_status = {
    "processed_count": 0,
    "total_count": 0,
    "status": "idle",  # idle, running, completed, error
    "error": None,
    "sample_results": [],
    "output_path": None
}

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
    global _extraction_status
    
    try:
        # Import here to avoid circular imports
        from business_workflow import run_extraction_workflow
        
        # Update status to running
        _extraction_status["status"] = "running"
        
        # Get total count from Excel file
        try:
            df = pd.read_excel(excel_path)
            _extraction_status["total_count"] = len(df)
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            _extraction_status["status"] = "error"
            _extraction_status["error"] = f"Error reading Excel file: {str(e)}"
            return ""
        
        # Run the workflow
        output_path = await run_extraction_workflow(excel_path, output_path=f"business_owners_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", max_concurrency=max_concurrency)
        
        # Read the results for a sample
        if os.path.exists(output_path):
            try:
                results_df = pd.read_csv(output_path)
                sample_size = min(5, len(results_df))
                sample_data = results_df.head(sample_size).to_dict(orient="records")
                
                _extraction_status["sample_results"] = sample_data
                _extraction_status["processed_count"] = len(results_df)
                _extraction_status["status"] = "completed"
                _extraction_status["output_path"] = output_path
                
            except Exception as e:
                logger.error(f"Error reading results file: {str(e)}")
                _extraction_status["status"] = "completed"
                _extraction_status["error"] = f"Process completed but error reading results: {str(e)}"
                _extraction_status["output_path"] = output_path
        else:
            _extraction_status["status"] = "error"
            _extraction_status["error"] = "Process completed but output file not found"
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error in extraction process: {str(e)}")
        _extraction_status["status"] = "error"
        _extraction_status["error"] = str(e)
        return ""

def start_extraction_process(excel_path: str, max_concurrency: int = 5) -> BusinessExtractionResults:
    """Start the extraction process as a background task."""
    global _extraction_task, _extraction_status
    
    # Reset status
    _extraction_status = {
        "processed_count": 0,
        "total_count": 0,
        "status": "starting",
        "error": None,
        "sample_results": [],
        "output_path": None
    }
    
    # Cancel any existing task
    if _extraction_task and not _extraction_task.done():
        logger.warning("Cancelling existing extraction task")
        # Note: We can't really cancel an asyncio task from another thread
        # but we'll mark the status as cancelled
        _extraction_status["status"] = "cancelled"
    
    # Create a new event loop for the background thread
    def run_async_extraction():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run_extraction_process(excel_path, max_concurrency))
        except Exception as e:
            logger.error(f"Error in background extraction thread: {str(e)}")
            _extraction_status["status"] = "error"
            _extraction_status["error"] = str(e)
        finally:
            loop.close()
    
    # Start the thread
    extraction_thread = threading.Thread(target=run_async_extraction)
    extraction_thread.daemon = True
    extraction_thread.start()
    
    # Return initial status
    return BusinessExtractionResults(**_extraction_status)

def get_extraction_status() -> BusinessExtractionResults:
    """Get the current status of the extraction process."""
    global _extraction_status
    return BusinessExtractionResults(**_extraction_status)

class ExtractBusinessInfoTool(BaseTool):
    """Tool for extracting business owner information from Excel file."""
    name: str = "extract_business_info"
    description: str = """
    Extracts business owner names and primary addresses for businesses in an Excel file.
    Uses advanced multi-query generation, web search, content crawling and RAG techniques.
    All processing is done asynchronously for efficient handling of large datasets.
    """
    args_schema: Type[BusinessExtractionInput] = BusinessExtractionInput
    
    def _run(self, excel_path: str, max_concurrency: int = 5) -> Dict[str, Any]:
        """Run the business information extraction process."""
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