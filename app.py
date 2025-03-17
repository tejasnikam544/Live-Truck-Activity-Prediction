from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib  # For loading the trained model

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")  # Ensure this is the correct trained model

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get JSON data from user input
        features = pd.DataFrame([data])  # Convert to DataFrame for prediction

        # Make prediction
        activity = model.predict(features)[0]

        return jsonify({"activity": activity})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
