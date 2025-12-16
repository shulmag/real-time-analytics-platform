# Created by Gil on 1/9/2025 for Blockchain server, but might help with all other servers too: 
    
# Use slim variant of Python image
# IMPROVEMENT: Reduced from ~1GB (full) to ~150MB (slim)
# WHY: Smaller image means faster deployments, smaller attack surface, lower costs
FROM python:3.10-slim

# Set environment variables
# PYTHONUNBUFFERED: Prevents Python from buffering stdout/stderr
# APP_HOME: Standardizes the workspace directory
# PORT: Makes the port configurable and explicit
# IMPROVEMENT: Added PORT env var for clarity
ENV PYTHONUNBUFFERED=True \
    APP_HOME=/app \
    PORT=8080

# Set working directory
# WHY: Keep files organized in a known location
WORKDIR $APP_HOME

# Install system dependencies
# IMPROVEMENT: Added essential build tools in same layer as cleanup
# WHY: Some Python packages need compilation, but we clean up immediately to keep image small
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements file separately
# IMPROVEMENT: Separated requirements copy from rest of code
# WHY: Better cache utilization - dependencies only reinstall if requirements.txt changes
COPY requirements.txt .

# Install Python dependencies
# IMPROVEMENT: Dependencies install in separate layer
# WHY: Better caching, faster builds when only application code changes
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# IMPROVEMENT: Copied after installing dependencies
# WHY: Changes to application code won't trigger reinstalling dependencies
COPY . .

# Create and switch to non-root user
# IMPROVEMENT: Added non-root user
# WHY: Security best practice - don't run containers as root
RUN useradd -m appuser && \
    chown -R appuser:appuser $APP_HOME
USER appuser

# Expose port
# IMPROVEMENT: Uses PORT from env var
# WHY: More explicit, follows 12-factor app principles
EXPOSE ${PORT}

# Start the application
# IMPROVEMENT: 
# 1. Single worker with multiple threads (vs multiple workers)
# 2. Added gthread worker class
# 3. Used explicit line continuation for readability
# WHY: 
# - Cloud Run handles horizontal scaling, so multiple workers per container aren't needed
# - Single worker with threads is more memory efficient
# - gthread worker class better handles Python's async workloads
CMD exec gunicorn --bind :${PORT} \
    --workers 1 \
    --threads 8 \
    --timeout 0 \
    --worker-class gthread \
    main:app