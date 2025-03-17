from flask import Flask, render_template, jsonify
import pandas as pd
import time
import threading
import joblib  # Load trained model

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")  # Ensure the correct model file is provided
df = pd.read_csv("data.csv")  # Load dataset
df = df.drop(columns=["Unnamed: 0"], errors="ignore")  # Drop unnecessary columns

# Initialize variables
current_index = 0
prediction_result = {"activity": "Waiting", "time_spent": {"Loading": 0, "Hauling": 0, "Unloading": 0, "Idling": 0}}
lock = threading.Lock()

def update_prediction():
    global current_index, prediction_result
    while True:
        with lock:
            if current_index < len(df):
                row = df.iloc[current_index]
                features = row.to_numpy().reshape(1, -1)  # Convert row to numpy array
                activity = model.predict(features)[0]  # Predict activity

                # Update time spent
                prediction_result["activity"] = activity
                prediction_result["time_spent"][activity] += 10  # Increment time (10 sec interval)

                current_index += 1
            else:
                prediction_result["activity"] = "Completed"

        time.sleep(10)  # Mock real-time prediction (10 sec interval)

# Run the prediction in a separate thread
threading.Thread(target=update_prediction, daemon=True).start()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    with lock:
        return jsonify(prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
