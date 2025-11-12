import os
import io
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# --- 0. MODEL DEFINITION (Same Architecture Used During Retraining) ---
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
            nn.Linear(50176, num_classes)
        )

    def forward(self, x):
        return self.sequential_model(x)


# --- 1. MODEL CONFIGURATION ---
MODELS = {}
CLASSES = {
    'xray': ['NORMAL', 'PNEUMONIA'],
    'mri': ['no', 'yes'],
    'ct': [
        'adenocarcinoma',
        'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
        'large.cell.carcinoma',
        'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
        'normal',
        'squamous.cell.carcinoma',
        'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
    ]
}

# Transformation same as retraining phase
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_models():
    """Loads only retrained models."""
    global MODELS
    device = torch.device("cpu")

    model_files = {
        'xray': 'model_xray_retrained_final_arch.pth',
        'mri': 'model_mri_retrained_final_arch.pth',
        'ct': 'model_ct_retrained_final_arch.pth'
    }

    for modality, model_file in model_files.items():
        if not os.path.exists(model_file):
            print(f"⚠️  {model_file} not found, skipping {modality}.")
            continue

        try:
            num_classes = len(CLASSES[modality])
            model = FinalSimpleCNN(num_classes=num_classes).to(device)
            state_dict = torch.load(model_file, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            MODELS[modality] = model
            print(f"✅ Loaded {modality.upper()} retrained model: {model_file}")
        except Exception as e:
            print(f"❌ Error loading {modality} model: {e}")

# --- 2. FASTAPI SETUP ---
app = FastAPI(title="AI Medical Analysis Backend (Retrained Models)")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    load_models()
    print("🚀 Models loaded successfully at startup")

# --- 3. API ENDPOINTS ---
@app.get("/")
def home():
    return {"message": "Backend is running with retrained models ✅"}

@app.post("/api/analyze/{modality}")
async def analyze_image(modality: str, file: UploadFile = File(...)):
    """Analyze X-ray, MRI, or CT images using retrained models."""
    if modality not in MODELS:
        return JSONResponse(status_code=400, content={"error": f"Invalid modality '{modality}'. Use xray, mri, or ct."})

    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = data_transform(image).unsqueeze(0)

        device = torch.device("cpu")
        model = MODELS[modality].to(device)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_index = torch.argmax(probabilities).item()
        diagnosis = CLASSES[modality][predicted_index]
        confidence = float(probabilities[predicted_index].item())

        # Generate report
        if diagnosis.upper() in ['NORMAL', 'NO']:
            status = "Good Health"
            anomalies = []
        else:
            status = "Diseases Detected"
            anomalies = [
                {
                    "name": f"Primary Anomaly: {diagnosis}",
                    "description": f"High confidence ({confidence:.2f}) detection of {diagnosis}. Further follow-up recommended."
                },
                {"name": "Secondary Findings", "description": "Review nearby areas for possible spread or misclassification."}
            ]

        return {
            "modality": modality,
            "diagnosis": status,
            "predicted_class": diagnosis,
            "confidence": confidence,
            "anomalies": anomalies
        }

    except Exception as e:
        print(f"❌ Error analyzing {modality}: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred during analysis."})


@app.get("/api/reload-models")
def reload_models():
    """Reload retrained models without restarting the backend."""
    load_models()
    return {"status": "✅ Models reloaded successfully."}
