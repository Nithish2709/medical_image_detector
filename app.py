from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io

# ============================================================
# 1️⃣ App Configuration
# ============================================================

app = FastAPI(title="Medical Image Anomaly Detector API")

# Allow all origins (safe for Hugging Face Space)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 2️⃣ Model Definition
# ============================================================

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

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.sequential_model(x)

# ============================================================
# 3️⃣ Image Preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ============================================================
# 4️⃣ Model Loading Logic
# ============================================================

MODELS = {}

def load_model(model_path, modality):
    try:
        model = SimpleCNN(num_classes=2)
        state_dict = torch.load(model_path, map_location="cpu")

        # Handle mismatched layer names
        if "sequential_model.10.weight" in state_dict:
            new_state_dict = {}
            for key, val in state_dict.items():
                new_key = key.replace("10", "9") if "10" in key else key
                new_state_dict[new_key] = val
            state_dict = new_state_dict

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"✅ Successfully loaded {modality.upper()} model from: {model_path}")
        return model

    except Exception as e:
        print(f"❌ Failed to load {modality.upper()} model:", e)
        return None


def load_all_models():
    global MODELS
    MODELS["xray"] = load_model("model_xray_retrained_final_arch.pth", "xray")
    MODELS["mri"] = load_model("model_mri_retrained_final_arch.pth", "mri")
    MODELS["ct"] = load_model("model_ct_retrained_final_arch.pth", "ct")
    print(f"📊 Loaded models: {list(MODELS.keys())}")

load_all_models()

# ============================================================
# 5️⃣ Inference Endpoint
# ============================================================

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...), modality: str = Form(...)):
    if modality not in MODELS or MODELS[modality] is None:
        return JSONResponse(status_code=400, content={"error": f"Invalid or unloaded modality: {modality}"})

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        model = MODELS[modality]
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        # class 0 = Normal, class 1 = Abnormal
        if predicted_class == 0:
            prediction = "Normal"
            recommendation = "No visible anomaly detected. Continue regular check-ups."
        else:
            prediction = "Abnormal"
            recommendation = "Potential anomaly detected. Please consult a specialist."

        return {
            "predicted_label": prediction,
            "confidence": f"{100 * probs[0][predicted_class]:.2f}%",
            "suggestion": recommendation,
            "modality": modality
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ============================================================
# 6️⃣ Root Endpoint
# ============================================================

@app.get("/")
def root():
    return {"message": "✅ Medical Image Anomaly Detector API Running Successfully!"}
