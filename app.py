from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io

app = FastAPI(title="Medical Image Anomaly Detector API")

# ======== Allow frontend requests ========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== CNN Architecture (matches retrained .pth) ========
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.sequential_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),   # Adjust based on your training input size
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.sequential_model(x)

# ======== Load all retrained models ========
MODELS = {}
class_names = ["Normal", "Abnormal"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model(path):
    model = SimpleCNN(num_classes=2)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_models():
    global MODELS
    try:
        print("🔄 Loading XRAY model...")
        MODELS["xray"] = load_model("model_xray_retrained_final_arch.pth")
        print("✅ XRAY model loaded.")

        print("🔄 Loading MRI model...")
        MODELS["mri"] = load_model("model_mri_retrained_final_arch.pth")
        print("✅ MRI model loaded.")

        print("🔄 Loading CT model...")
        MODELS["ct"] = load_model("model_ct_retrained_final_arch.pth")
        print("✅ CT model loaded.")

        print(f"📊 Final MODELS keys: {list(MODELS.keys())}")
        print("🚀 All models loaded successfully.")
    except Exception as e:
        print("❌ Model loading failed:", e)

load_models()

# ======== Inference ========
def predict_image(image: Image.Image, model):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    return class_names[pred_idx], confidence

# ======== Endpoint ========
@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...), modality: str = Form(...)):
    if modality not in MODELS:
        return JSONResponse(status_code=400, content={"error": f"Invalid modality: {modality}"})

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model = MODELS[modality]
        pred_label, confidence = predict_image(image, model)

        suggestion = (
            "No visible anomaly detected."
            if pred_label == "Normal"
            else "Potential anomaly detected. Please consult a specialist."
        )

        print(f"[{modality.upper()}] Prediction: {pred_label} ({confidence:.3f})")

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
