#!/bin/bash

# Exit on any error
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Wait for any initialization if needed
echo "Starting Medical Report Automation API..."

# Run the application
exec "$@"
