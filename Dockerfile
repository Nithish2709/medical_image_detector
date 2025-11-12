# Use a Python base image with necessary libraries
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- COPY ALL FILES FROM THE FLAT REPOSITORY STRUCTURE (ROOT) ---
# Copy the FastAPI application file (named app.py in your repo)
COPY app.py .

# Copy the model files from the repository root
COPY model_ct_retrained_final_arch.pth .
COPY model_mri_retrained_final_arch.pth .
COPY model_xray_retrained_final_arch.pth .

# Set the environment variable for Python path
ENV PYTHONPATH=/app

# Expose the port (Hugging Face standard)
EXPOSE 7860

# Command to run the application using uvicorn
# 'app:app' refers to the 'app' object inside 'app.py' located in the /app directory.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]