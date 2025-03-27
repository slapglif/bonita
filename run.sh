#!/bin/bash

echo "Starting application with Python 3.11..."
export PYTHONPATH=/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages:/home/runner/workspace
python3.11 -m gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app