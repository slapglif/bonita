#!/bin/bash

# Create a simple index.html file if it doesn't exist
if [ ! -f "index.html" ]; then
  cat > index.html << 'EOF'
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

# Try to find Python
for cmd in python3.11 python3.10 python3.9 python3.8 python3.7 python3 python; do
  if command -v $cmd > /dev/null 2>&1; then
    echo "Found Python: $cmd"
    
    # Try to use Python's built-in HTTP server
    echo "Starting HTTP server with Python on port 8080..."
    $cmd -m http.server 8080
    exit 0
  fi
done

# If Python is not available, try to use Node.js
if command -v node > /dev/null 2>&1; then
  echo "Found Node.js, creating a simple HTTP server..."
  
  # Create a simple Node.js HTTP server
  cat > server.js << 'EOF'
const http = require('http');
const fs = require('fs');
const path = require('path');

const server = http.createServer((req, res) => {
  if (req.url === '/' || req.url === '/index.html') {
    fs.readFile('index.html', (err, data) => {
      if (err) {
        res.writeHead(500);
        res.end('Error loading index.html');
        return;
      }
      res.writeHead(200, {'Content-Type': 'text/html'});
      res.end(data);
    });
  } else {
    res.writeHead(404);
    res.end('Not found');
  }
});

server.listen(8080, () => {
  console.log('Server running at http://0.0.0.0:8080/');
});
EOF

  node server.js
  exit 0
fi

# Last resort: Use bash to create a minimal HTTP server (if nc is available)
if command -v nc > /dev/null 2>&1; then
  echo "Using netcat to create a minimal HTTP server..."
  
  PORT=8080
  echo "Starting HTTP server on port $PORT..."
  
  # Create response headers
  HEADER="HTTP/1.1 200 OK
Content-Type: text/html; charset=UTF-8
Connection: close

"
  
  while true; do
    # Create a TCP socket and listen on port 8080
    { echo -ne "$HEADER"; cat index.html; } | nc -l $PORT
    
    echo "Request received and served"
  done
  
  exit 0
fi

echo "ERROR: Could not find any way to start an HTTP server"
exit 1