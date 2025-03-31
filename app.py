import os
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler

# # Initialize Flask app
# app = Flask(__name__)

from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- MODEL 1: Skin and Presoil Classification (VGG16) ----------
def load_model_pth(model_path, default_classes=9):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    checkpoint = torch.load(model_path, map_location=device)
    
    class_names = checkpoint.get("class_names", None)
    num_classes = checkpoint["state_dict"]["classifier.6.weight"].shape[0] if "state_dict" in checkpoint else default_classes
    
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_features, num_classes)
    
    state_dict_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
    model.load_state_dict(checkpoint[state_dict_key])

    if class_names is None:
        class_names = [f"Condition_{i}" for i in range(num_classes)]

    model.to(device)
    model.eval()
    return model, class_names

# Preprocessing function for images
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Prediction function for images
def predict_image(image, model, class_names):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = output.max(1)
    return class_names[predicted.item()]

# Load models for classification
skin_model_path = "skin_cancer_vgg16_cpu_model.pth"
presoil_model_path = "model (1).pth"

skin_model, skin_class_names = load_model_pth(skin_model_path)
presoil_model, presoil_class_names = load_model_pth(presoil_model_path)

# Endpoint for skin disease prediction
@app.route("/predict_skin", methods=["POST"])
def predict_skin():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    prediction = predict_image(image, skin_model, skin_class_names)
    
    return jsonify({"predicted_condition": prediction})

# Endpoint for presoil classification
@app.route("/predict_presoil", methods=["POST"])
def predict_presoil():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    prediction = predict_image(image, presoil_model, presoil_class_names)
    
    return jsonify({"predicted_class": prediction})

# ---------- MODEL 2: Soil Fertility Prediction (SoilNet) ----------
class SoilNet(nn.Module):
    def __init__(self):
        super(SoilNet, self).__init__()
        self.fc1 = nn.Linear(14, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load soil fertility model and scalers
soil_model = SoilNet()
soil_model.load_state_dict(torch.load('soil_model.pth', map_location=device))
soil_model.eval()

X_scaler = torch.load('soil_X_scaler.pth')
Y_scaler = torch.load('soil_Y_scaler.pth')

# Nutrient thresholds for analysis
NUTRIENT_THRESHOLDS = {
    'NO3': 12.75, 'P': 47, 'K': 15, 'Organic Matter': 0.28,
    'Fe': 1, 'Zn': 0.6, 'pH': 6.5
}

def analyze_soil_nutrients(current_nutrients):
    low_nutrients = [nutrient for nutrient, threshold in NUTRIENT_THRESHOLDS.items() if current_nutrients[nutrient] < threshold]

    if not low_nutrients:
        return "Your soil has sufficient levels of all major nutrients."
    
    analysis = "Soil Nutrient Analysis:\n\n"
    analysis += "The following nutrients are below optimal levels:\n"

    for nutrient in low_nutrients:
        current_value = current_nutrients[nutrient]
        threshold = NUTRIENT_THRESHOLDS[nutrient]
        analysis += f"- {nutrient}: Current level ({current_value}) is below threshold ({threshold})\n"

    return analysis
@app.route('/soilfertility', methods=['POST'])
def soilfertility():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate input
        if not data or len(data) != 14:
            return jsonify({"error": "Invalid input data. Expecting 14 values."})

        # Convert JSON data to a list of floats
        try:
            inputs = [float(data[str(i)]) for i in range(14)]
        except (ValueError, KeyError):
            return jsonify({"error": "Invalid data format. Ensure all 14 fields are present and numerical."})

        if all(x == 0 for x in inputs):
            return jsonify({"prediction": 0, "status": "No valid input data"})

        current_nutrients = {
            'NO3': inputs[0], 'NH4': inputs[1], 'P': inputs[2], 'K': inputs[3],
            'SO4': inputs[4], 'B': inputs[5], 'Organic Matter': inputs[6], 'pH': inputs[7],
            'Zn': inputs[8], 'Cu': inputs[9], 'Fe': inputs[10], 'Ca': inputs[11],
            'Mg': inputs[12], 'Na': inputs[13]
        }

        # Scale input and predict
        input_scaled = X_scaler.transform([inputs])
        input_tensor = torch.FloatTensor(input_scaled)
        with torch.no_grad():
            prediction = soil_model(input_tensor).numpy()[0][0]

        # Determine fertility status
        if prediction < 0.3:
            status = "Bad"
        elif prediction < 0.7:
            status = "Less Fertile"
        else:
            status = "High Fertile"

        # Nutrient analysis
        nutrient_analysis = analyze_soil_nutrients(current_nutrients) if status in ["Bad", "Less Fertile"] else "Soil is healthy."

        return jsonify({
            "prediction": round(prediction * 100),
            "status": status,
            "nutrient_analysis": nutrient_analysis
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[""],  # Replace "" with specific domains if needed
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods
#     allow_headers=["*"],  # Allow all headers
# )

# @app.get("/")
# def read_root():
#     return {"message": "Hello World"}


# import os
# import io
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from PIL import Image
# from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sklearn.preprocessing import MinMaxScaler

# # Initialize FastAPI app
# app = FastAPI()

# CORS(app)



# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ---------- MODEL 1: Skin and Presoil Classification (VGG16) ----------
# def load_model_pth(model_path, default_classes=9):
#     model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
#     checkpoint = torch.load(model_path, map_location=device)
    
#     class_names = checkpoint.get("class_names", None)
#     num_classes = checkpoint["state_dict"]["classifier.6.weight"].shape[0] if "state_dict" in checkpoint else default_classes
    
#     num_features = model.classifier[6].in_features
#     model.classifier[6] = torch.nn.Linear(num_features, num_classes)
    
#     state_dict_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
#     model.load_state_dict(checkpoint[state_dict_key])

#     if class_names is None:
#         class_names = [f"Condition_{i}" for i in range(num_classes)]

#     model.to(device)
#     model.eval()
#     return model, class_names

# # Preprocessing function for images
# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     return transform(image).unsqueeze(0).to(device)

# # Prediction function for images
# def predict_image(image, model, class_names):
#     image_tensor = preprocess_image(image)
#     with torch.no_grad():
#         output = model(image_tensor)
#         _, predicted = output.max(1)
#     return class_names[predicted.item()]

# # Load models for classification
# skin_model_path = "skin_cancer_vgg16_cpu_model.pth"
# presoil_model_path = "model (1).pth"

# skin_model, skin_class_names = load_model_pth(skin_model_path)
# presoil_model, presoil_class_names = load_model_pth(presoil_model_path)

# @app.post("/predict_skin")
# async def predict_skin(file: UploadFile = File(...)):
#     image = Image.open(io.BytesIO(await file.read())).convert("RGB")
#     prediction = predict_image(image, skin_model, skin_class_names)
#     return {"predicted_condition": prediction}

# @app.post("/predict_presoil")
# async def predict_presoil(file: UploadFile = File(...)):
#     image = Image.open(io.BytesIO(await file.read())).convert("RGB")
#     prediction = predict_image(image, presoil_model, presoil_class_names)
#     return {"predicted_class": prediction}

# # ---------- MODEL 2: Soil Fertility Prediction (SoilNet) ----------
# class SoilNet(nn.Module):
#     def __init__(self):
#         super(SoilNet, self).__init__()
#         self.fc1 = nn.Linear(14, 32)
#         self.fc2 = nn.Linear(32, 16)
#         self.fc3 = nn.Linear(16, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # Load soil fertility model and scalers
# soil_model = SoilNet()
# soil_model.load_state_dict(torch.load('soil_model.pth', map_location=device))
# soil_model.eval()

# X_scaler = torch.load('soil_X_scaler.pth')
# Y_scaler = torch.load('soil_Y_scaler.pth')

# # Nutrient thresholds for analysis
# NUTRIENT_THRESHOLDS = {
#     'NO3': 12.75, 'P': 47, 'K': 15, 'Organic Matter': 0.28,
#     'Fe': 1, 'Zn': 0.6, 'pH': 6.5
# }

# class SoilInput(BaseModel):
#     values: list[float]

# @app.post('/soilfertility')
# async def soilfertility(data: SoilInput):
#     inputs = data.values
#     if len(inputs) != 14:
#         raise HTTPException(status_code=400, detail="Invalid input data. Expecting 14 values.")
    
#     if all(x == 0 for x in inputs):
#         return {"prediction": 0, "status": "No valid input data"}
    
#     current_nutrients = {
#         'NO3': inputs[0], 'NH4': inputs[1], 'P': inputs[2], 'K': inputs[3],
#         'SO4': inputs[4], 'B': inputs[5], 'Organic Matter': inputs[6], 'pH': inputs[7],
#         'Zn': inputs[8], 'Cu': inputs[9], 'Fe': inputs[10], 'Ca': inputs[11],
#         'Mg': inputs[12], 'Na': inputs[13]
#     }
    
#     input_scaled = X_scaler.transform([inputs])
#     input_tensor = torch.FloatTensor(input_scaled)
#     with torch.no_grad():
#         prediction = soil_model(input_tensor).numpy()[0][0]
    
#     status = "High Fertile" if prediction >= 0.7 else "Less Fertile" if prediction >= 0.3 else "Bad"
    
#     if status in ["Bad", "Less Fertile"]:
#         analysis = "Soil Nutrient Analysis:\n\n"
#         low_nutrients = [nutrient for nutrient, threshold in NUTRIENT_THRESHOLDS.items() if current_nutrients[nutrient] < threshold]
#         for nutrient in low_nutrients:
#             analysis += f"- {nutrient}: Below optimal\n"
#     else:
#         analysis = "Soil is healthy."
    
#     return {"prediction": round(prediction * 100), "status": status, "nutrient_analysis": analysis}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=3000)
