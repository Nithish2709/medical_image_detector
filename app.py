import os
import io
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# ==============================
# 1️⃣ Model Architecture
# ==============================
class FinalSimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(FinalSimpleCNN, self).__init__()
        self.sequential_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, num_classes)  # Adjust depending on your input size
        )

    def forward(self, x):
        return self.sequential_model(x)


# ==============================
# 2️⃣ Model Configurations
# ==============================
CLASSES = {
    "xray": ["NORMAL", "PNEUMONIA"],
    "mri": ["no", "yes"],
    "ct": [
        "adenocarcinoma", "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib",
        "large.cell.carcinoma", "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
        "normal", "squamous.cell.carcinoma",
        "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"
    ]
}

MODEL_PATHS = {
    "xray": "model_xray_retrained_final_arch.pth",
    "mri": "model_mri_retrained_final_arch.pth",
    "ct": "model_ct_retrained_final_arch.pth"
}

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

MODELS = {}


# ==============================
# 3️⃣ Model Loading Function
# ==============================
def load_models():
    global MODELS
    device = torch.device("cpu")

    for modality, path in MODEL_PATHS.items():
        try:
            print(f"🔄 Loading {modality.upper()} model from: {path}")

            num_classes = len(CLASSES[modality])
            model = FinalSimpleCNN(num_classes=num_classes)
            state_dict = torch.load(path, map_location=device)

            # Load model weights
            model.load_state_dict(state_dict)
            model.eval()

            MODELS[modality] = model
            print(f"✅ Successfully loaded {modality.upper()} retrained model.")
        except Exception as e:
            print(f"❌ Failed to load {modality.upper()} model: {e}")

    print("📊 Final MODELS dictionary keys:", list(MODELS.keys()))


# ==============================
# 4️⃣ FastAPI Setup
# ==============================
app = FastAPI(title="Medical Image Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    load_models()
    print("🚀 Models loaded successfully at startup")


# ==============================
# 5️⃣ API Endpoint
# ==============================
@app.post("/api/analyze/{modality}")
async def analyze_image(modality: str, file: UploadFile = File(...)):
    if modality not in MODELS:
        return JSONResponse(status_code=400, content={"error": f"Model type '{modality}' not recognized."})

    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = data_transform(image).unsqueeze(0)

        device = torch.device("cpu")
        model = MODELS[modality].to(device)
        input_tensor = input_tensor.to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_index = torch.argmax(probabilities).item()
        diagnosis = CLASSES[modality][predicted_index]
        confidence = probabilities[predicted_index].item()

        # Prepare response
        if diagnosis.upper() in ["NORMAL", "NO"]:
            status = "Good Health"
            anomalies = []
        else:
            status = "Diseases Detected"
            anomalies = [
                {"name": f"Primary Anomaly: {diagnosis}", "description": f"Detected with confidence {confidence:.2f}"},
                {"name": "Recommendation", "description": "Please consult a specialist for further examination."}
            ]

        return {
            "modality": modality,
            "diagnosis": status,
            "predicted_class": diagnosis,
            "confidence": confidence,
            "anomalies": anomalies
        }

    except Exception as e:
        print(f"⚠️ Error analyzing {modality}: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error during analysis."})


# ==============================
# 6️⃣ Health Check
# ==============================
@app.get("/")
async def root():
    return {"message": "Medical Image Analysis API is running successfully!"}
