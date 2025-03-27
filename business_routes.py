"""
Business Extraction Routes

This module provides Flask routes for interacting with the business information extraction workflow.
"""
import os
import asyncio
import logging
import re
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, send_file

from business_processor import start_extraction_process, get_extraction_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("business_routes")

# Create Blueprint
business_bp = Blueprint('business', __name__, url_prefix='/api/business')

@business_bp.route('/extract', methods=['POST'])
def extract_business_info():
    """Start the business information extraction process."""
    # Check request format
    if not request.is_json:
        logger.error("Invalid request format: expected JSON")
        return jsonify({
            "error": "Invalid request format. Please send a JSON request.",
            "status": "error"
        }), 400
    
    # Check API keys with detailed error messages
    missing_keys = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.environ.get("SERPAPI_API_KEY"):
        missing_keys.append("SERPAPI_API_KEY")
    
    if missing_keys:
        error_msg = f"Missing required API keys: {', '.join(missing_keys)}"
        logger.error(error_msg)
        return jsonify({
            "error": error_msg,
            "status": "error",
            "missing_keys": missing_keys,
            "message": "Please configure the required API keys before extraction."
        }), 400
    
    # Process the request and handle errors
    try:
        # Get parameters with validation
        excel_path = request.json.get('excel_path')
        if not excel_path:
            return jsonify({
                "error": "Missing required parameter: excel_path",
                "status": "error"
            }), 400
        
        # Get optional parameters with defaults
        max_concurrency = None
        if 'max_concurrency' in request.json:
            try:
                max_concurrency = int(request.json.get('max_concurrency'))
                if max_concurrency < 1:
                    return jsonify({
                        "error": "max_concurrency must be a positive integer",
                        "status": "error"
                    }), 400
            except ValueError:
                return jsonify({
                    "error": "max_concurrency must be a valid integer",
                    "status": "error"
                }), 400
        
        # Check that Excel file exists
        if not os.path.exists(excel_path):
            return jsonify({
                "error": f"Excel file not found: {excel_path}",
                "status": "error",
                "file_path": excel_path,
                "message": "Please provide a valid Excel file path."
            }), 404
        
        # Check that the file is an Excel file
        if not excel_path.lower().endswith(('.xlsx', '.xls')):
            return jsonify({
                "error": f"Invalid file format: {excel_path}. Only Excel files (.xlsx, .xls) are supported.",
                "status": "error"
            }), 400
        
        # Start the extraction process
        logger.info(f"Starting extraction process for {excel_path} with max_concurrency={max_concurrency}")
        result = start_extraction_process(excel_path, max_concurrency)
        
        # Return success response with detailed information
        return jsonify({
            "status": "started",
            "message": f"Processing started for {excel_path}",
            "details": result.dict(),
            "total_businesses": result.total_count
        })
    
    except Exception as e:
        logger.error(f"Error starting extraction: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "error",
            "message": "An unexpected error occurred while starting the extraction process."
        }), 500

@business_bp.route('/status', methods=['GET'])
def check_extraction_status():
    """Check the current status of the extraction process."""
    try:
        # Get the current status
        status = get_extraction_status()
        
        # Enhance the response with additional information
        response = status.dict()
        
        # Add human-readable status descriptions
        status_descriptions = {
            "not_started": "The extraction process has not been started yet.",
            "in_progress": "The extraction process is currently running.",
            "completed": "The extraction process has completed successfully.",
            "failed": "The extraction process encountered an error and could not complete."
        }
        
        # Add the description if available
        if response.get("status") in status_descriptions:
            response["status_description"] = status_descriptions[response["status"]]
        
        # Calculate progress percentage if possible
        if response.get("processed_count") is not None and response.get("total_count") is not None and response["total_count"] > 0:
            response["progress_percentage"] = round((response["processed_count"] / response["total_count"]) * 100, 2)
        else:
            response["progress_percentage"] = 0.0
        
        # Add timestamp
        response["timestamp"] = datetime.now().isoformat()
        
        # Add API key status (without exposing the keys)
        response["api_keys_configured"] = {
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
            "serpapi": bool(os.environ.get("SERPAPI_API_KEY"))
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error checking extraction status: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "error",
            "message": "An unexpected error occurred while checking the extraction status.",
            "timestamp": datetime.now().isoformat()
        }), 500

@business_bp.route('/download/<path:filename>', methods=['GET'])
def download_results(filename):
    """Download extraction results."""
    try:
        # Validate the filename to prevent directory traversal attacks
        import os.path
        import re
        
        # Ensure the filename doesn't contain path traversal components
        if '..' in filename or filename.startswith('/') or not re.match(r'^[\w\-. /]+$', filename):
            logger.warning(f"Suspicious download path requested: {filename}")
            return jsonify({
                "error": "Invalid filename format",
                "status": "error",
                "message": "The requested file path is not allowed for security reasons."
            }), 403
        
        # Only allow CSV and Excel files to be downloaded
        allowed_extensions = ['.csv', '.xlsx', '.xls']
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            logger.warning(f"Attempted to download non-business data file: {filename}")
            return jsonify({
                "error": f"Invalid file type. Only {', '.join(allowed_extensions)} files can be downloaded.",
                "status": "error"
            }), 403
        
        # Check if file exists
        if not os.path.exists(filename):
            logger.error(f"Requested download file not found: {filename}")
            return jsonify({
                "error": f"File not found: {filename}",
                "status": "error",
                "message": "The requested file does not exist. Make sure extraction has completed."
            }), 404
        
        # Check if extraction status is complete
        status = get_extraction_status()
        if status.status != "completed" and status.output_path == filename:
            logger.warning(f"Attempted to download incomplete results: {filename}")
            return jsonify({
                "error": "Extraction in progress",
                "status": "error",
                "message": "The extraction process is still in progress. Please wait for completion before downloading results.",
                "current_status": status.dict()
            }), 409
        
        # Get file info for the response
        import os.path
        
        file_stats = os.stat(filename)
        file_size = file_stats.st_size
        file_modified = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        
        # Add custom headers for better download experience
        filename_base = os.path.basename(filename)
        
        # Log the successful download
        logger.info(f"Sending file for download: {filename} ({file_size} bytes)")
        
        # Send the file with appropriate headers
        response = send_file(
            filename, 
            as_attachment=True,
            download_name=filename_base,
            mimetype="text/csv" if filename.endswith('.csv') else "application/vnd.ms-excel"
        )
        
        # Add custom headers
        response.headers["X-File-Size"] = str(file_size)
        response.headers["X-File-Modified"] = file_modified
        
        return response
    
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "error",
            "message": "An unexpected error occurred while trying to download the file."
        }), 500