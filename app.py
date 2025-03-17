from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import time
from threading import Thread

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Global variables to store uploaded data and predictions
uploaded_data = None
predictions = []
total_time = 0
current_index = 0

# Function to simulate real-time predictions
def process_data_real_time():
    global current_index, predictions, total_time
    while current_index < len(uploaded_data):
        row = uploaded_data.iloc[current_index, :].values.reshape(1, -1)
        row_scaled = scaler.transform(row)  # Scale the input
        predicted_activity = model.predict(row_scaled)[0]  # Predict activity
        
        predictions.append({
            'Time': (current_index + 1) * 10,
            'Activity': predicted_activity
        })
        
        current_index += 1
        time.sleep(10)  # Simulate real-time arrival every 10s
    
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_data, predictions, total_time, current_index
    file = request.files['file']
    if file:
        uploaded_data = pd.read_csv(file)
        uploaded_data = uploaded_data.iloc[:, 1:]  # Remove unnecessary columns
        total_time = len(uploaded_data) * 10
        predictions = []
        current_index = 0
        
        # Start real-time processing in a separate thread
        thread = Thread(target=process_data_real_time)
        thread.start()
        return jsonify({'message': 'File uploaded successfully, processing started'})
    return jsonify({'error': 'No file uploaded'})

@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    return jsonify({'predictions': predictions, 'total_time': total_time})

if __name__ == '__main__':
    app.run(debug=True)
