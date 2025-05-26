#!/usr/bin/env bash

# Setup Python environment
export PYTHONPATH="$(pwd)"

# Run the application with the specified configuration
python3 app/main.py --config "$1"
