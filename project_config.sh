#!/bin/bash

# Toggle debug messages
DEBUG=${DEBUG:-false} 

# Set ROOT_DIR to the directory containing this script
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
export ROOT_DIR="$SCRIPT_DIR"

# Optional debug output
if [ "$DEBUG" = true ]; then
    echo "[DEBUG] ROOT_DIR set to: $ROOT_DIR"
fi

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false
