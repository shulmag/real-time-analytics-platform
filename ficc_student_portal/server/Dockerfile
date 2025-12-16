# Use Python 3.9 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Set environment variables
ENV PORT=8080

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}