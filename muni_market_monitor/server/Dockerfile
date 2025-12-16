# Use slim variant of Python image for smaller size
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=True \
    APP_HOME=/app \
    PORT=8080

# Set working directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy local code to container
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE $PORT

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT 