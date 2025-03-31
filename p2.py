import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, flash, render_template, url_for, redirect
from sklearn.preprocessing import MinMaxScaler

# Define the neural network model
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

# Load the saved model and scalers
model = SoilNet()
model.load_state_dict(torch.load('soil_model.pth'))
model.eval()  # Set the model to evaluation mode
X_scaler = torch.load('soil_X_scaler.pth')
Y_scaler = torch.load('soil_Y_scaler.pth')

data = pd.read_csv('dataset.txt')
app = Flask(__name__)

# Define nutrient thresholds
NUTRIENT_THRESHOLDS = {
    'NO3': 12.75,  # ppm
    'P': 47,       # ppm
    'K': 15,       # ppm
    'Organic Matter': 0.28,  # percentage
    'Fe': 1,       # ppm
    'Zn': 0.6,     # ppm
    'pH': 6.5      # pH level
}

def analyze_soil_nutrients(current_nutrients):
    # Check which nutrients are below thresholds
    low_nutrients = []
    for nutrient, threshold in NUTRIENT_THRESHOLDS.items():
        if current_nutrients[nutrient] < threshold:
            low_nutrients.append(nutrient)
    
    if not low_nutrients:
        return "Your soil has sufficient levels of all major nutrients."
    
    analysis = "Soil Nutrient Analysis:\n\n"
    analysis += "The following nutrients are below optimal levels:\n"
    
    for nutrient in low_nutrients:
        current_value = current_nutrients[nutrient]
        threshold = NUTRIENT_THRESHOLDS[nutrient]
        analysis += f"- {nutrient}: Current level ({current_value}) is below threshold ({threshold})\n"
    
    # Add pH analysis
    ph_value = current_nutrients['pH']
    if ph_value < 6.0:
        analysis += "\nSoil pH is too acidic (below 6.0). This can affect nutrient availability."
    elif ph_value > 7.5:
        analysis += "\nSoil pH is too alkaline (above 7.5). This can affect nutrient availability."
    
    # Add organic matter analysis
    om_value = current_nutrients['Organic Matter']
    if om_value < 0.28:
        analysis += "\nLow organic matter content indicates poor soil structure and nutrient retention."
    
    return analysis

@app.route('/')
def home():
    return render_template('ProjectHomepage.html')

@app.route('/soilfertility', methods=['POST'])
def soilfertility():
    # Get form data
    a0 = float(request.form['0'])
    a1 = float(request.form['1'])
    a2 = float(request.form['2'])
    a3 = float(request.form['3'])
    a4 = float(request.form['4'])
    a5 = float(request.form['5'])
    a6 = float(request.form['6'])
    a7 = float(request.form['7'])
    a8 = float(request.form['8'])
    a9 = float(request.form['9'])
    a10 = float(request.form['10'])
    a11 = float(request.form['11'])
    a12 = float(request.form['12'])
    a13 = float(request.form['13'])

    if all(x == 0 for x in [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13]):
        return render_template('ProjectHomepage.html', prediction_text=0)

    # Prepare data for prediction
    current_nutrients = {
        'NO3': a0, 'NH4': a1, 'P': a2, 'K': a3, 'SO4': a4, 'B': a5,
        'Organic Matter': a6, 'pH': a7, 'Zn': a8, 'Cu': a9, 'Fe': a10,
        'Ca': a11, 'Mg': a12, 'Na': a13
    }
    
    # Create input array for prediction
    input_data = np.array([[a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13]])
    
    # Scale the input data
    input_scaled = X_scaler.transform(input_data)
    
    # Convert to PyTorch tensor
    input_tensor = torch.FloatTensor(input_scaled)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor).numpy()[0][0]
    
    # Determine current soil fertility status
    current_status = ""
    if prediction < 0.3:
        current_status = "Bad"
    elif prediction < 0.7:
        current_status = "Less Fertile"
    else:
        current_status = "High Fertile"

    # Get nutrient analysis
    nutrient_analysis = ""
    if current_status in ["Bad", "Less Fertile"]:
        nutrient_analysis = analyze_soil_nutrients(current_nutrients)

    return render_template('Results.html',
                         current_status=current_status,
                         prediction_text=np.round(prediction*100).astype(int),
                         recommendations=nutrient_analysis)

if __name__ == "__main__":
    app.run(debug=True)
