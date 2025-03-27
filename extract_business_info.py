#!/usr/bin/env python
"""
Business Information Extraction CLI

This script runs the business information extraction workflow from a command-line interface,
allowing users to process an Excel file and extract business owner information.
"""
import os
import sys
import asyncio
import logging
import argparse
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_api_keys() -> bool:
    """Check that required API keys are present."""
    missing_keys = []
    
    if not os.environ.get("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.environ.get("SERPAPI_API_KEY"):
        missing_keys.append("SERPAPI_API_KEY")
    
    if missing_keys:
        logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
        return False
    
    return True

async def main(excel_path: str, output_path: Optional[str] = None, max_concurrency: int = 5) -> int:
    """Run the business information extraction workflow."""
    try:
        # Import business_workflow here to ensure all dependencies are properly loaded
        from business_workflow import run_extraction_workflow
        
        # Check API keys
        if not check_api_keys():
            return 1
        
        # Default output path if not provided
        if not output_path:
            output_path = "business_owners_output.csv"
        
        logger.info(f"Starting extraction workflow from '{excel_path}' with concurrency {max_concurrency}")
        logger.info(f"Results will be saved to '{output_path}'")
        
        # Run the extraction workflow
        result_path = await run_extraction_workflow(excel_path, output_path)
        
        if os.path.exists(result_path):
            logger.info(f"Extraction completed successfully. Results saved to: {result_path}")
            return 0
        else:
            logger.error("Extraction failed - output file was not created")
            return 1
            
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        logger.error("Make sure all dependencies are installed")
        return 1
    except Exception as e:
        logger.error(f"Extraction process failed: {str(e)}")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract business owner information from Excel file")
    parser.add_argument("excel_path", help="Path to Excel file containing business data")
    parser.add_argument("--output", "-o", help="Path to save CSV output (default: business_owners_output.csv)")
    parser.add_argument("--concurrency", "-c", type=int, default=5, 
                        help="Maximum number of businesses to process concurrently (default: 5)")
    
    args = parser.parse_args()
    
    # Run the async main function
    exit_code = asyncio.run(main(args.excel_path, args.output, args.concurrency))
    sys.exit(exit_code)