from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import random
import time

app = Flask(__name__)

# Simulated real-time truck activity data
truck_activities = [
    {"time": i * 10, "activity": random.choice(["Loading", "Hauling", "Unloading", "Idling"])}
    for i in range(20)
]

@app.route("/")
def home():
    return render_template("home.html", data=truck_activities)

@app.route("/get_predictions")
def get_predictions():
    """Simulates real-time truck activity updates."""
    new_data = {"time": truck_activities[-1]["time"] + 10, "activity": random.choice(["Loading", "Hauling", "Unloading", "Idling"])}
    truck_activities.append(new_data)
    
    return jsonify(new_data)

if __name__ == "__main__":
    app.run(debug=True)
