#!/bin/bash

# Get the current directory
ROOT_DIR=$(pwd)

# Print directory structure
echo "=== PROJECT DIRECTORY STRUCTURE ==="
tree -I 'node_modules|venv|.git' .
echo

# Loop through all regular files, excluding .pkl, hidden files/dirs, and common large dirs
find "$ROOT_DIR" -type f \
  -not -path "*/\.*" \
  -not -path "*/node_modules/*" \
  -not -path "*/venv/*" \
  -not -path "*/.git/*" \
  -not -name "*.pkl" \
  -not -name "*.txt" \
  -not -name "*.csv" \
  -not -name "*.duckdb" \
  -not -name "*.wal" \
  -not -path "/lolpredictor/__pycache__/*" \
  -not -name "*.json" \
  -not -name "*.cpython-312.pyc" \
  | while read -r file; do
    echo "=== $file ==="
    cat "$file"
    echo "+++END $file+++"
    echo
done
