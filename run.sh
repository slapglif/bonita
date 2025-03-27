#!/bin/bash

# Find any python in the path
PYTHON_PATH=$(find / -name "python3.11" 2>/dev/null | head -1)

if [ -z "$PYTHON_PATH" ]; then
  PYTHON_PATH=$(find / -name "python3" 2>/dev/null | head -1)
fi

if [ -z "$PYTHON_PATH" ]; then
  PYTHON_PATH=$(find / -name "python" 2>/dev/null | head -1)
fi

echo "Found Python at: $PYTHON_PATH"

if [ -z "$PYTHON_PATH" ]; then
  echo "Error: Could not find Python"
  exit 1
fi

# Run the application
echo "Running Python application..."
$PYTHON_PATH main.py