#!/bin/bash

# Minimal web server using netcat
echo "Starting minimal web server on port 5000..."

# Create HTML content
cat > index.html << 'EOF'
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
    </div>
</body>
</html>
EOF

# Serve HTTP responses using a loop
while true; do
  echo -e "HTTP/1.1 200 OK\nContent-Type: text/html\n" > response.txt
  cat index.html >> response.txt
  nc -l -p 5000 < response.txt
done