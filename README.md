---
title: Medical Image Anomaly Detector
---

# Medical Image Anomaly Detector API 🏥⚕️

A machine learning-powered API that detects medical anomalies from X-ray, MRI, and CT scans using Convolutional Neural Networks (CNNs). 

**Theme:** HealthTech  
**Tech Stack:** Python, FastAPI, PyTorch, Docker  

## 🚀 Overview

This project provides a REST API to analyze medical images and predict potential health issues. The custom CNN models have been trained to detect:
- **X-Ray:** Pneumonia
- **MRI:** Brain Tumors
- **CT Scans:** Lung Cancer

## 📂 Project Structure

- `app.py`: The FastAPI application containing the API endpoints, CNN model architecture, and inference logic.
- `Dockerfile`: Configuration for containerizing the application.
- `requirements.txt`: Python dependencies.
- `.pth` files: Pre-trained PyTorch model weights for X-ray, MRI, and CT scan analysis.

## 🛠️ Setup & Installation

### Option 1: Running with Docker (Recommended)

1. Build the Docker image:
   ```bash
   docker build -t medical-image-detector .
   ```
2. Run the container:
   ```bash
   docker run -p 7860:7860 medical-image-detector
   ```
   *(Note: The default port for Hugging Face Spaces is 7860, adjust if running locally)*

### Option 2: Running Locally

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
   The API will be accessible at `http://localhost:8000`. You can view the interactive documentation at `http://localhost:8000/docs`.

## 📡 API Endpoints

### `GET /`
Health check endpoint. Returns a welcome message, list of available modalities, and the supported class labels.

### `POST /api/analyze`
Analyzes an uploaded medical image and returns the diagnosis.

**Request Parameters:**
- `file` (File): The medical image file to be analyzed.
- `modality` (Form): The type of scan. Must be one of `xray`, `mri`, or `ct`.

**Example Response:**
```json
{
  "predicted_label": "Pneumonia Detected",
  "confidence": "98.45%",
  "suggestion": "Signs of pneumonia detected. Consult a pulmonologist for further evaluation.",
  "modality": "XRAY"
}
```

## ⚠️ Disclaimer
This tool is for educational and informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.

