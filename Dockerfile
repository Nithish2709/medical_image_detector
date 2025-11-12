# Use a Python base image with all necessary standard libraries
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- COPY ALL PROJECT FILES ---
COPY app.py .
COPY model_ct_retrained_final_arch.pth .
COPY model_mri_retrained_final_arch.pth .
COPY model_xray_retrained_final_arch.pth .

# Add environment variable for Python path
ENV PYTHONPATH=/app

# Expose port 7860 (used by Hugging Face)
EXPOSE 7860

# Start FastAPI server using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
