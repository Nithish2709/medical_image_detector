import os
import io
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# --- 0. DEDUCED MODEL ARCHITECTURE ---
class FinalSimpleCNN(nn.Module):
    # ... (Model definition remains the same) ...
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

# --- 1. MODEL CONFIGURATION AND LOADING ---
MODELS = {}
CLASSES = {
    'xray': ['NORMAL', 'PNEUMONIA'],
    'mri': ['no', 'yes'],
    'ct': [
        'adenocarcinoma', 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 
        'large.cell.carcinoma', 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 
        'normal', 'squamous.cell.carcinoma', 
        'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
    ]
}

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_models():
    """Load models for all three modalities from the root directory."""
    global MODELS
    device = torch.device("cpu") 
    
    for specialty, classes in CLASSES.items():
        # *** FIX: Model path is now relative to the container root /app ***
        model_filename = f"model_{specialty}_retrained_final_arch.pth"
        model_path = os.path.join(".", model_filename) # Looks in /app/model_*.pth
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Skipping.")
            continue
            
        model = FinalSimpleCNN(num_classes=len(classes)).to(device)
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=True) 
            model.eval()
            MODELS[specialty] = model
            print(f"Loaded {specialty} model successfully.")
        except Exception as e:
            print(f"Error loading {specialty} model: {e}")

# --- 2. FASTAPI SETUP & CORS ---
app = FastAPI()

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
    """FIX: Automatic loading is enabled here to make the API functional."""
    load_models()
    print("FastAPI started. Models loaded on startup event.")


@app.post("/api/analyze/{modality}")
async def analyze_image(modality: str, file: UploadFile = File(...)):
    """API endpoint to receive image, run inference, and return analysis."""
    # ... (Inference logic remains the same) ...
    if not MODELS:
        return JSONResponse(status_code=503, content={"error": "Models not loaded."})
        
    if modality not in MODELS:
        return JSONResponse(status_code=400, content={"error": f"Analysis for {modality} not supported."})

    try:
        # 1. Read Image and Preprocess (rest of the logic...)
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        input_tensor = data_transform(image)
        input_batch = input_tensor.unsqueeze(0) 

        device = torch.device("cpu")
        model = MODELS[modality].to(device)
        input_batch = input_batch.to(device)
        
        with torch.no_grad():
            output = model(input_batch)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_index = torch.argmax(probabilities).item()
        diagnosis = CLASSES[modality][predicted_index]
        
        # Determine diagnosis status based on class name
        if diagnosis.upper() in ['NORMAL', 'NO']:
            status = 'Good Health'
            anomalies = []
        else:
            status = 'Diseases Detected'
            anomalies = [
                {"name": f"Primary Anomaly: {diagnosis}", "description": f"High confidence diagnosis for {diagnosis}. Requires urgent follow-up."},
                {"name": "Secondary Findings", "description": "Recommend reviewing adjacent slices for confirmation."},
            ]

        return {
            "modality": modality,
            "diagnosis": status,
            "predicted_class": diagnosis,
            "confidence": probabilities[predicted_index].item(),
            "anomalies": anomalies
        }

    except Exception as e:
        print(f"Analysis failed for {modality}: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred during model analysis. Check logs."})