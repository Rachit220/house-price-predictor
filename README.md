# 🏠 House Price Predictor — ML Project

A complete Machine Learning web app using **Random Forest Regressor** to predict house prices.
No dataset download needed — everything is auto-generated!

---

## 🛠️ Tech Stack

| Layer       | Technology                       |
|-------------|----------------------------------|
| ML Model    | Random Forest (scikit-learn)     |
| Backend     | Flask (Python)                   |
| Frontend    | HTML + CSS + Chart.js            |
| Language    | Python 3.9+                      |
| Platform    | Runs locally on your machine     |

---

## 📁 Project Structure

```
house_price_predictor/
│
├── train_model.py       ← Train the ML model
├── app.py               ← Flask web server
├── requirements.txt     ← Python dependencies
│
├── templates/
│   └── index.html       ← Beautiful web UI
│
├── data/                ← Auto-created after training
│   └── housing_data.csv
│
└── models/              ← Auto-created after training
    ├── model.pkl
    └── scaler.pkl
```

---

## 🚀 How to Run (Step by Step)

### Step 1 — Install Python
Download Python 3.9+ from https://python.org  
✅ Check "Add to PATH" during install (Windows)

### Step 2 — Open Terminal
- **Windows**: Press Win+R → type `cmd` → Enter
- **Mac/Linux**: Open Terminal app

### Step 3 — Navigate to Project Folder
```bash
cd path/to/house_price_predictor
```

### Step 4 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Train the Model
```bash
python train_model.py
```
You'll see output like:
```
🔧 Generating dataset …
   ✅ 2000 rows saved to data/housing_data.csv
🤖 Training Random Forest …
📊 Model Performance:
   MAE : $12,345
   R²  : 0.9712
✅ model.pkl & scaler.pkl saved to models/
```

### Step 6 — Launch the Web App
```bash
python app.py
```

### Step 7 — Open in Browser
Go to: **http://127.0.0.1:5000**

🎉 You're done! Adjust the sliders and predict house prices!

---

## 🧠 How the ML Works

1. **Data**: 2000 synthetic houses with 8 features
2. **Features**: Size, Bedrooms, Bathrooms, Age, Garage, Location, Floors, Pool
3. **Algorithm**: Random Forest (200 decision trees)
4. **Preprocessing**: StandardScaler normalization
5. **Evaluation**: MAE + R² score on 20% test split

---

## 🖥️ Platform Compatibility

| Platform | Status |
|----------|--------|
| Windows  | ✅ Works |
| macOS    | ✅ Works |
| Linux    | ✅ Works |
| Google Colab | ⚠️ Train only (no Flask UI) |

---

## 💡 Customize It

- Change `n_estimators` in `train_model.py` for more accuracy
- Modify `n_samples=2000` for larger dataset
- Edit `templates/index.html` to change the UI
- Replace synthetic data with a real CSV (Kaggle, etc.)
