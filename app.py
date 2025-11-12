from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io

app = FastAPI(title="Medical Image Anomaly Detector API")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (or restrict to your HF space)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== Load Models ========

MODELS = {}

def load_models():
    global MODELS
    try:
        print("🔄 Loading XRAY model from: model_xray_retrained_final_arch.pth")
        MODELS["xray"] = torch.load("model_xray_retrained_final_arch.pth", map_location="cpu")
        print("✅ Successfully loaded XRAY retrained model.")

        print("🔄 Loading MRI model from: model_mri_retrained_final_arch.pth")
        MODELS["mri"] = torch.load("model_mri_retrained_final_arch.pth", map_location="cpu")
        print("✅ Successfully loaded MRI retrained model.")

        print("🔄 Loading CT model from: model_ct_retrained_final_arch.pth")
        MODELS["ct"] = torch.load("model_ct_retrained_final_arch.pth", map_location="cpu")
        print("✅ Successfully loaded CT retrained model.")

        print(f"📊 Final MODELS dictionary keys: {list(MODELS.keys())}")
        print("🚀 Models loaded successfully at startup")
    except Exception as e:
        print("❌ Model loading failed:", e)

load_models()

# ======== ROUTE ========

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...), modality: str = Form(...)):
    if modality not in MODELS:
        return JSONResponse(status_code=400, content={"error": f"Invalid modality: {modality}"})

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Dummy inference example (replace with actual PyTorch inference)
        predicted_label = "Normal" if "normal" in file.filename.lower() else "Abnormal"
        suggestion = (
            "No visible anomaly detected" if predicted_label == "Normal"
            else "Potential anomaly detected. Please consult a specialist."
        )

        return {
            "predicted_label": predicted_label,
            "suggestion": suggestion,
            "modality": modality
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "Medical Image Anomaly Detector API Running"}
