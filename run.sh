#!/bin/bash
# Launch script for the Government Scheme RAG Assistant
# Usage:
#   ./run.sh          - Start the web UI (default)
#   ./run.sh voice    - Start the terminal voice assistant
#   ./run.sh web      - Start the web UI explicitly

set -e

CHATBOT_DIR=~/programs/sandyproj/chatbotrag
VENV=~/programs/ComfyUI/compyy

MODE="${1:-web}"

if [ "$MODE" = "voice" ]; then
    echo "=== Starting terminal voice assistant ==="
    echo "    (whisper.cpp port 8080, llama.cpp port 8081)"
    $VENV/bin/python $CHATBOT_DIR/chatbot.py
else
    echo "=== Starting web UI ==="
    echo "    Chat:    http://localhost:5000/"
    echo "    Admin:   http://localhost:5000/admin"
    echo "    Default password: admin123"
    echo ""
    echo "    Set ADMIN_PASSWORD env var to change password."
    echo ""
    $VENV/bin/python $CHATBOT_DIR/webapp.py
fi
