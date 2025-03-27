import http.server
import socketserver

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Minimal Business Extraction App</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
                    .container { max-width: 800px; margin: 0 auto; background-color: #fff; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                    h1 { color: #333; }
                    .alert { padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 5px; margin-bottom: 15px; }
                    .note { padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px; margin-bottom: 15px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Business Information Extraction System</h1>
                    <div class="note">
                        <p><strong>Status:</strong> Application is running in minimal mode</p>
                    </div>
                    <p>This is a minimal version of the Business Information Extraction System.</p>
                    <p>The full version requires the following packages to be installed:</p>
                    <ul>
                        <li>Flask</li>
                        <li>Pandas</li>
                        <li>LangChain</li>
                        <li>LangGraph</li>
                        <li>OpenAI</li>
                        <li>Playwright</li>
                        <li>SerpAPI</li>
                    </ul>
                    <p>Once these packages are installed, the full version will be available.</p>

                    <h2>System Features:</h2>
                    <ul>
                        <li><strong>High Performance Processing:</strong> Extracts business owner information with maximum parallelism</li>
                        <li><strong>Multi-Query Generation:</strong> Creates diverse search strategies for each business</li>
                        <li><strong>Dynamic Search Expansion:</strong> Leverages React agent's observations to expand searches when necessary</li>
                        <li><strong>Comprehensive Analysis:</strong> Provides detailed confidence scores and reasoning paths</li>
                        <li><strong>Web Content Crawling:</strong> Accesses and processes web content from search results</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            self.wfile.write(html_content.encode())
        else:
            # Serve files from the current directory
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

if __name__ == "__main__":
    PORT = 5000
    handler = MyHandler
    
    print(f"Starting server on port {PORT}...")
    with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
        print(f"Server running at http://0.0.0.0:{PORT}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped by user")