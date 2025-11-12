from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import io

app = FastAPI(title="Medical Image Anomaly Detector API")

# ======== Allow Frontend Requests ========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== Model Setup ========

class_names = ["Normal", "Abnormal"]  # must match your retraining dataset order

# Common preprocessing (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

MODELS = {}

def build_model(num_classes=2):
    """Define the same architecture used during retraining."""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_models():
    global MODELS
    try:
        print("🔄 Loading XRAY model from: model_xray_retrained_final_arch.pth")
        xray_model = build_model()
        xray_model.load_state_dict(torch.load("model_xray_retrained_final_arch.pth", map_location="cpu"))
        xray_model.eval()
        MODELS["xray"] = xray_model
        print("✅ Successfully loaded XRAY retrained model.")

        print("🔄 Loading MRI model from: model_mri_retrained_final_arch.pth")
        mri_model = build_model()
        mri_model.load_state_dict(torch.load("model_mri_retrained_final_arch.pth", map_location="cpu"))
        mri_model.eval()
        MODELS["mri"] = mri_model
        print("✅ Successfully loaded MRI retrained model.")

        print("🔄 Loading CT model from: model_ct_retrained_final_arch.pth")
        ct_model = build_model()
        ct_model.load_state_dict(torch.load("model_ct_retrained_final_arch.pth", map_location="cpu"))
        ct_model.eval()
        MODELS["ct"] = ct_model
        print("✅ Successfully loaded CT retrained model.")

        print(f"📊 Final MODELS dictionary keys: {list(MODELS.keys())}")
        print("🚀 Models loaded successfully at startup")
    except Exception as e:
        print("❌ Model loading failed:", e)

load_models()

# ======== Inference Function ========
def predict_image(image: Image.Image, model, class_names):
    """Run inference on an input image."""
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    return class_names[pred_idx], confidence

# ======== API Route ========

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...), modality: str = Form(...)):
    if modality not in MODELS:
        return JSONResponse(status_code=400, content={"error": f"Invalid modality: {modality}"})

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model = MODELS[modality]
        pred_label, confidence = predict_image(image, model, class_names)

        suggestion = (
            "No visible anomaly detected."
            if pred_label == "Normal"
            else "Potential anomaly detected. Please consult a specialist."
        )

        return {
            "predicted_label": pred_label,
            "confidence": round(confidence, 3),
            "suggestion": suggestion,
            "modality": modality
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "Medical Image Anomaly Detector API Running"}
