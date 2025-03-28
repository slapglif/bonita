"""
Main application entry point

This module serves as the entry point for the application,
starting the HTTP server with the configured app.
"""
import http.server
import socketserver
import os

# Use SimpleHTTPRequestHandler to serve static files
handler = http.server.SimpleHTTPRequestHandler

if __name__ == "__main__":
    PORT = 8080
    
    print(f"Starting server on port {PORT}...")
    with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
        print(f"Server running at http://0.0.0.0:{PORT}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped by user")
