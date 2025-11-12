# Use a Python base image with necessary libraries
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the directory structure needed for the app
RUN mkdir -p /app/app/models

# Copy the Python application code and the models directory
COPY app/main.py /app/app/
COPY app/models/ /app/app/models/

# Copy the startup script, ensure it's executable
COPY start.sh /app/
RUN chmod +x /app/start.sh

# Set the environment variable for Python path
ENV PYTHONPATH=/app

# Expose the port (Hugging Face standard for apps)
EXPOSE 7860

# Command to run the application using the startup script
CMD ["/app/start.sh"]