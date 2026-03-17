"""
🏠 House Price Predictor - Flask Web App
"""

from flask import Flask, render_template, request, jsonify
import pickle, numpy as np, os

app = Flask(__name__)

MODEL_PATH  = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model, scaler
    if model is None:
        model, scaler = load_artifacts()
    if model is None:
        return jsonify({"error": "Model not trained yet. Run train_model.py first!"})

    data = request.get_json()
    features = np.array([[
        float(data["size"]),
        int(data["bedrooms"]),
        int(data["bathrooms"]),
        int(data["age"]),
        int(data["garage"]),
        int(data["location"]),
        int(data["floors"]),
        int(data["pool"]),
    ]])
    scaled   = scaler.transform(features)
    price    = model.predict(scaled)[0]

    # Feature importances for the chart
    importances = model.feature_importances_.tolist()
    feature_names = [
        "Size", "Bedrooms", "Bathrooms", "Age",
        "Garage", "Location", "Floors", "Pool"
    ]

    return jsonify({
        "price": round(float(price), 2),
        "importances": importances,
        "feature_names": feature_names,
    })

if __name__ == "__main__":
    print("🚀 Starting House Price Predictor …")
    print("   Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)
