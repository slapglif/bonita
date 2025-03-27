"""
Main application entry point

This module serves as the entry point for the application,
starting the Flask server with the configured app.
"""
from app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
