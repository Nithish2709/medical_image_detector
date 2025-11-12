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
# This class must match the architecture of the models saved (e.g., model_xray.pth)
class FinalSimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(FinalSimpleCNN, self).__init__()
        # Architecture deduced from the layer keys: "0", "3", "7", "9"
        self.sequential_model = nn.Sequential(
            # Key 0: Conv layer
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Key 3: Conv layer
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Key 7: Conv layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Flatten(),
            
            # Key 9: Final Linear layer (Input size: 28*28*64 = 50176)
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
        'adenocarcinoma', 
        'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 
        'large.cell.carcinoma', 
        'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 
        'normal', 
        'squamous.cell.carcinoma', 
        'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
    ]
}

# Preprocessing pipeline
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_models():
    """Load models for all three modalities."""
    global MODELS
    # Use CPU since Hugging Face Spaces free tier typically uses CPU
    device = torch.device("cpu") 
    
    # Model files are assumed to be in the 'models' directory inside the app container
    for specialty, classes in CLASSES.items():
        # Using the saved '_retrained_final_arch.pth' name
        model_path = os.path.join("app", "models", f"model_{specialty}_retrained_final_arch.pth") 
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Skipping.")
            continue
            
        model = FinalSimpleCNN(num_classes=len(classes)).to(device)
        try:
            state_dict = torch.load(model_path, map_location=device)
            # Use strict=True if loading the final saved models with the matching FinalSimpleCNN architecture
            model.load_state_dict(state_dict, strict=True) 
            model.eval()
            MODELS[specialty] = model
            print(f"Loaded {specialty} model successfully.")
        except Exception as e:
            print(f"Error loading {specialty} model: {e}")

# The line calling load_models() has been removed here to allow manual loading.
# If you are controlling the model loading outside of the FastAPI startup sequence,
# ensure this load_models() function is called before the /api/analyze endpoint is hit.

# --- 2. FASTAPI SETUP & CORS ---
app = FastAPI()

# IMPORTANT: CORS Configuration
# Allows all origins (*) to connect to the API, essential for your local frontend
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
    """
    Hugging Face Spaces automatically calls this event on startup.
    Since the model loading is manual/external, we leave load_models() commented out here.
    The start.sh script will execute it externally.
    """
    # load_models()
    print("FastAPI started. Models are set for manual or external loading.")


@app.post("/api/analyze/{modality}")
async def analyze_image(modality: str, file: UploadFile = File(...)):
    """API endpoint to receive image, run inference, and return analysis."""
    # Check if models have been loaded manually
    if not MODELS:
        return JSONResponse(status_code=503, content={"error": "Models not loaded. Please ensure manual model loading is complete."})
        
    if modality not in MODELS:
        return JSONResponse(status_code=400, content={"error": f"Analysis for {modality} not supported or model failed to load."})

    try:
        # 1. Read Image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 2. Preprocess
        input_tensor = data_transform(image)
        input_batch = input_tensor.unsqueeze(0) 

        # 3. Inference
        device = torch.device("cpu")
        model = MODELS[modality].to(device)
        input_batch = input_batch.to(device)
        
        with torch.no_grad():
            output = model(input_batch)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # 4. Process Results
        predicted_index = torch.argmax(probabilities).item()
        
        diagnosis = CLASSES[modality][predicted_index]
        
        # Determine diagnosis status based on class name
        if diagnosis.upper() in ['NORMAL', 'NO']:
            status = 'Good Health'
            anomalies = []
        else:
            status = 'Diseases Detected'
            # Return detailed anomalies based on the specific diagnosis
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
        # Log the error for debugging
        print(f"Analysis failed for {modality}: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred during model analysis. Check logs."})