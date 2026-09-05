#!/bin/bash
# Test script for the multilingual voice assistant
# Assumes whisper.cpp (port 8080) and llama.cpp (port 8081) are already running

set -e

CHATBOT_DIR=~/programs/sandyproj/chatbotrag
VENV=~/programs/ComfyUI/compyy

echo "=== Starting chatbot ==="
$VENV/bin/python $CHATBOT_DIR/chatbot.py
