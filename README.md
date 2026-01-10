# Docker Setup for Medical Report Automation API

This document provides instructions for running the Medical Report Automation API using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (usually comes with Docker Desktop)

## Quick Start

### 1. Environment Setup

First, create your environment file:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file and add your Anthropic API key
# ANTHROPIC_API_KEY=your_actual_api_key_here
```

### 2. Build and Run with Docker Compose

```bash
# Build and start the application
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

### 3. Access the Application

- **API Documentation**: http://localhost:8000/swaggers
- **Health Check**: http://localhost:8000/swaggers (used for health checks)

## Manual Docker Commands

### Build the Docker Image

```bash
docker build -t medical-poc .
```

### Run the Container

```bash
# Run with environment variables
docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY=your_api_key_here \
  medical-poc

# Or run with .env file
docker run -p 8080:8080 --env-file .env medical-poc
```

## Environment Variables

```
ANTHROPIC_API_KEY = your-anthropic-api-key
```

## Docker Compose Features

- **Health Checks**: Automatic health monitoring
- **Volume Mounting**: Logs are persisted to `./logs` directory
- **Network**: Isolated network for the application
- **Restart Policy**: Automatically restarts on failure
- **Environment**: Loads from `.env` file

## Useful Commands

### View Logs

```bash
# View logs from docker-compose
docker-compose logs -f

# View logs from specific service
docker-compose logs -f medical-poc
```

### Stop the Application

```bash
# Stop and remove containers
docker-compose down

# Stop, remove containers, and remove volumes
docker-compose down -v
```

### Rebuild After Changes

```bash
# Rebuild and restart
docker-compose up --build

# Force rebuild without cache
docker-compose build --no-cache
```

### Access Container Shell

```bash
# Get shell access to running container
docker-compose exec medical-poc /bin/bash

# Or with docker run
docker run -it --env-file .env medical-poc /bin/bash
```


### Viewing Application Logs

```bash
# Real-time logs
docker-compose logs -f medical-poc

# Last 100 lines
docker-compose logs --tail=100 medical-poc
```




## Run with Uvicorn
```
uvicorn app.main:app --reload
```

