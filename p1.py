import torch
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
import io

# Initialize Flask app
app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load model
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

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Prediction function
def predict_image(image, model, class_names):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = output.max(1)
    return class_names[predicted.item()]

# Load both models
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
