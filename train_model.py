"""
🏠 House Price Predictor - ML Model Training
=============================================
Algorithm: Random Forest Regressor
Dataset: Synthetic (auto-generated, no download needed!)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os


# 1. GENERATE SYNTHETIC DATASET

def generate_dataset(n_samples=2000, seed=42):
    np.random.seed(seed)

    size        = np.random.randint(500, 5000, n_samples)
    bedrooms    = np.random.randint(1, 7, n_samples)
    bathrooms   = np.random.randint(1, 5, n_samples)
    age         = np.random.randint(0, 50, n_samples)
    garage      = np.random.randint(0, 4, n_samples)
    location    = np.random.randint(1, 5, n_samples)   # 1=rural … 4=prime
    floors      = np.random.randint(1, 4, n_samples)
    pool        = np.random.randint(0, 2, n_samples)

    # Formula: realistic price generation
    price = (
        size        * 120
        + bedrooms  * 15_000
        + bathrooms * 12_000
        - age       * 800
        + garage    * 10_000
        + location  * 30_000
        + floors    * 8_000
        + pool      * 25_000
        + np.random.normal(0, 20_000, n_samples)   # noise
    )
    price = np.clip(price, 50_000, 2_000_000)

    df = pd.DataFrame({
        "Size_sqft":  size,
        "Bedrooms":   bedrooms,
        "Bathrooms":  bathrooms,
        "Age_years":  age,
        "Garage_cars": garage,
        "Location_score": location,
        "Floors":     floors,
        "Has_Pool":   pool,
        "Price":      price.astype(int),
    })
    return df
# 2. TRAIN
def train():
    print("🔧 Generating dataset …")
    df = generate_dataset()
    df.to_csv("data/housing_data.csv", index=False)
    print(f"   ✅ {len(df)} rows saved to data/housing_data.csv")

    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\n🤖 Training Random Forest …")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"\n📊 Model Performance:")
    print(f"   MAE : ${mae:,.0f}")
    print(f"   R²  : {r2:.4f}")

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl",  "wb") as f: pickle.dump(model,  f)
    with open("models/scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    print("\n✅ model.pkl & scaler.pkl saved to models/")
    return mae, r2

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    train()
    print("\n🎉 Training complete! Run `python app.py` next.")
