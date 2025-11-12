from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI(title="Medical Image Analysis API")

# --- Allow frontend requests ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load retrained models ---
MODEL_PATHS = {
    "xray": "model_xray_retrained_final_arch.pth",
    "mri": "model_mri_retrained_final_arch.pth",
    "ct": "model_ct_retrained_final_arch.pth"
}

MODELS = {}

for model_type, path in MODEL_PATHS.items():
    try:
        model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
        MODELS[model_type] = model
        print(f"✅ Loaded {model_type.upper()} retrained model: {path}")
    except Exception as e:
        print(f"❌ Failed to load {model_type.upper()} model: {e}")

print("🚀 Models loaded successfully at startup")


# --- Image transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# --- API routes ---

@app.get("/")
async def root():
    return {"message": "Medical Image Analysis Backend is Running 🚀"}


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...), model_type: str = "xray"):
    """
    API endpoint to analyze an uploaded image using retrained model.
    - `file`: the uploaded medical image (X-ray, MRI, or CT)
    - `model_type`: one of 'xray', 'mri', 'ct'
    """
    if model_type not in MODELS:
        return {"error": f"Model type '{model_type}' not recognized."}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = MODELS[model_type](img_tensor)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    return {
        "model_type": model_type,
        "prediction": int(prediction),
        "message": f"Prediction completed using {model_type.upper()} retrained model."
    }
