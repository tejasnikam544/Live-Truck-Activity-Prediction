from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib  # For loading ML models
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained model
model = joblib.load("model.pkl")  # Replace with your trained model

# Function to process data and predict activities
def predict_activity(file_path):
    df = pd.read_csv(file_path)
    
    # Assuming the necessary preprocessing is already handled
    features = df.drop(columns=["time"])  # Modify as needed
    predictions = model.predict(features)

    df["Predicted Activity"] = predictions

    # Calculate total time spent on each activity
    activity_summary = df["Predicted Activity"].value_counts().to_dict()

    return activity_summary

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Predict activity and get summary
    activity_summary = predict_activity(file_path)

    return jsonify({"activity_summary": activity_summary})

if __name__ == "__main__":
    app.run(debug=True)
