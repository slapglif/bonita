"""
Business Extraction Routes

This module provides Flask routes for interacting with the business information extraction workflow.
"""
import os
import asyncio
import logging
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
    # Check API keys
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("SERPAPI_API_KEY"):
        return jsonify({
            "error": "Missing required API keys. Please set OPENAI_API_KEY and SERPAPI_API_KEY."
        }), 400
    
    # Check that file exists
    try:
        excel_path = request.json.get('excel_path', 'sample.xlsx')
        max_concurrency = int(request.json.get('max_concurrency', 5))
        
        if not os.path.exists(excel_path):
            return jsonify({
                "error": f"Excel file not found: {excel_path}"
            }), 404
        
        # Start the extraction process
        result = start_extraction_process(excel_path, max_concurrency)
        
        return jsonify({
            "status": "started",
            "message": f"Processing started for {excel_path}",
            "details": result.dict()
        })
    
    except Exception as e:
        logger.error(f"Error starting extraction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@business_bp.route('/status', methods=['GET'])
def check_extraction_status():
    """Check the current status of the extraction process."""
    try:
        status = get_extraction_status()
        return jsonify(status.dict())
    
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@business_bp.route('/download/<path:filename>', methods=['GET'])
def download_results(filename):
    """Download extraction results."""
    try:
        # Check if file exists
        if not os.path.exists(filename):
            return jsonify({"error": f"File not found: {filename}"}), 404
        
        # Send the file
        return send_file(filename, as_attachment=True)
    
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({"error": str(e)}), 500