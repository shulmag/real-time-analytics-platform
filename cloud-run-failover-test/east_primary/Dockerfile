# Use the official Python image.
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY main.py ./

# Expose the port the app runs on
EXPOSE 8080

# Set the environment variable for Flask
ENV FLASK_APP=main.py

# Start the application
CMD ["python", "main.py"]