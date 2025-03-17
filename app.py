from flask import Flask, render_template, request, jsonify
import pandas as pd
import time
import threading
import eventlet
import os
from flask_socketio import SocketIO
import joblib

eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# Load trained model
model = joblib.load("model.pkl")  # Ensure your trained model is saved as model.pkl

# Activity time tracking
activity_time = {"Loading": 0, "Dumping": 0, "Idling": 0, "Hauling": 0}
time_interval = 10  # Each data point represents 10 seconds

uploaded_file_path = None

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_file_path
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)
    uploaded_file_path = file_path

    threading.Thread(target=process_file, args=(file_path,)).start()
    return jsonify({"message": "File uploaded and processing started"}), 200

def process_file(file_path):
    global activity_time
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        features = row.drop(["Unnamed: 0"])  # Drop unnecessary columns
        prediction = model.predict([features])[0]  # Predict activity

        # Update activity time
        activity_time[prediction] += time_interval / 60  # Convert to minutes

        # Send update to frontend
        socketio.emit('update', {'activity': prediction, 'activity_time': activity_time})
        time.sleep(10)  # Simulating real-time delay

@app.route('/reset', methods=['POST'])
def reset():
    global activity_time
    activity_time = {"Loading": 0, "Dumping": 0, "Idling": 0, "Hauling": 0}
    return jsonify({"message": "Activity time reset"}), 200

if __name__ == '__main__':
    socketio.run(app, debug=True)
