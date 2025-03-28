#!/bin/bash

# Simple HTTP server using Bash
# Listens on port 5000

PORT=5000
HTML_FILE="index.html"

# Check if index.html exists, if not create it
if [ ! -f "$HTML_FILE" ]; then
  cat > "$HTML_FILE" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Business Information Extraction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .features {
            margin-top: 20px;
        }
        .features li {
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Business Information Extraction</h1>
        <p>This application extracts business owner information from Excel files using advanced techniques.</p>
        
        <div class="features">
            <h2>Key Features:</h2>
            <ul>
                <li>Multi-query generation for comprehensive search</li>
                <li>Parallel processing with maximum concurrency</li>
                <li>Advanced content analysis with React-style reasoning</li>
                <li>Dynamic search expansion based on confidence thresholds</li>
            </ul>
        </div>
        
        <div class="status">
            <h2>System Status:</h2>
            <p>The API server is currently running and accepting requests for business information extraction.</p>
            <p>Use the API endpoints to upload Excel files and process business data.</p>
        </div>
    </div>
</body>
</html>
EOF
fi

echo "Starting HTTP server on port $PORT..."
echo "Use Ctrl+C to stop the server"

# Create response headers
HEADER="HTTP/1.1 200 OK
Content-Type: text/html; charset=UTF-8
Connection: close
"

while true; do
  # Create a TCP socket and listen on port 5000
  # Use nc -l for some systems or nc -l -p for others (like Ubuntu)
  { echo -ne "$HEADER"; cat "$HTML_FILE"; } | nc -l $PORT
  
  echo "Request received and served"
done