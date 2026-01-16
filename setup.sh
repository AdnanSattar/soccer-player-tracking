#!/bin/bash

echo "Soccer Tracker - UV Setup Script"
echo "===================================="

echo ""
echo "1. Checking if UV is installed..."
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    pip install uv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install UV"
        exit 1
    fi
else
    echo "UV is installed: $(uv --version)"
fi

echo ""
echo "2. Creating virtual environment..."
if [ -d ".venv" ]; then
    echo ".venv directory already exists. Skipping creation."
else
    uv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
fi

echo ""
echo "3. Installing project dependencies..."
source .venv/bin/activate
uv pip install -e .
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "===================================="
echo "Setup completed successfully!"
echo "===================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Then you can run the tracker with:"
echo "  python main.py"
echo "===================================="
