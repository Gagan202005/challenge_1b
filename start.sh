#!/bin/bash
# Start Ollama server in the background
ollama serve &
sleep 5

# Run your script with passed arguments (like --collection ...)
python run_rag.py "$@"