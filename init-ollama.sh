#!/bin/bash
echo "Waiting for Ollama to be ready..."
sleep 10

echo "Pulling phi3 model..."
ollama pull phi3

echo "Phi3 model installed successfully"