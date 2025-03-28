import http.server
import socketserver

PORT = 5000

handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
    print(f"Server running at http://0.0.0.0:{PORT}/")
    httpd.serve_forever()