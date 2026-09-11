# ✈️ British Airways Booking Prediction

A machine learning project that predicts whether a customer is likely to complete a flight booking based on booking behaviour, travel details, and customer preferences.

## 🚀 Live Demo

**Streamlit App:**
https://british-airways-booking-prediction.onrender.com

---

## 📌 Project Overview

This project analyzes customer booking behaviour using a 40,000+ record airline booking dataset and builds a classification model to predict booking completion.

### Key Work
- Data cleaning and exploratory data analysis (EDA)
- Feature engineering
- Categorical and target encoding
- Model comparison
- Imbalanced-class evaluation
- XGBoost model development
- Streamlit interactive user interface
- FastAPI REST API
- Docker containerization
- Render deployment

---

## 🤖 Machine Learning

Models evaluated:
- Logistic Regression
- Random Forest
- XGBoost

The final application uses **XGBoost**.

### Final Model Configuration

| Parameter | Value |
|---|---:|
| Model | XGBoost |
| Trees | 500 |
| Learning Rate | 0.05 |
| Max Depth | 5 |
| Scale Pos Weight | 5.6 |
| Evaluation Metric | Logloss |
| Decision Threshold | 0.52 |
| Test Accuracy | ~73% |

Because the target is imbalanced, performance was evaluated using accuracy, precision, recall, F1-score, ROC-AUC and PR-AUC — not accuracy alone.

---

## 🖥️ Streamlit Application

Users can enter customer, booking, travel, preference, origin and route information.

The app returns:
- Booking probability
- Booking prediction
- Decision threshold
- Business interpretation

Route and booking-origin values are available through dropdown menus using the saved training encoding maps.

---

## 🔌 FastAPI

The project also includes a FastAPI REST service.

**Endpoint:** `POST /predict`

Swagger documentation when running locally:
http://localhost:8000/docs

---

## 🧩 Feature Engineering

The prediction pipeline includes:

- Total service preference
- Additional service indicator
- Length-of-stay groups (0–7, 8–30, 31–60, 61–90, 91–120, 121–180, 181–250, 251–365, 365+ days)
- Purchase-lead groups (0–7, 8–30, 31–60, 61–120, 121+ days)
- Weekend-based flight-day feature
- Route target encoding
- Booking-origin target encoding
- One-hot encoding
- Feature alignment with training columns
- Scaling using the saved scaler

---

## 🐳 Docker

**Build:**
```bash
docker build -t ba-prediction-analysis-api .
```

**Run locally:**
```bash
docker run -p 8000:8000 ba-prediction-analysis-api
```

The deployed Streamlit container uses the hosting platform's `PORT` environment variable.

---

## 📁 Project Structure

```
British-Airways-booking-prediction/
├── BA_final_file.ipynb
├── BA_model.pkl
├── BA_scaler.pkl
├── BA_feature_columns.pkl
├── route_encoding_map.pkl
├── origin_encoding_map.pkl
├── global_target_mean.pkl
├── app.py
├── ui.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · FastAPI · Streamlit · Joblib · Docker · Render

---

## ▶️ Run Locally

```bash
git clone https://github.com/vishva-jeet/British-Airways-booking-prediction.git
cd British-Airways-booking-prediction

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run ui.py
```

To run the FastAPI service:
```bash
uvicorn app:app --reload
```

---

## 💼 Business Use Case

The model can help identify customers with different booking-conversion likelihoods and support:
- Targeted offers
- Personalized services
- Follow-up engagement
- Conversion-focused campaigns

---

## 📄 Dataset

The project uses a British Airways customer booking behaviour dataset containing 40,000+ records.

The raw dataset is excluded from the repository; trained models and preprocessing artifacts are included for inference.

---

## 👤 Author

**Vishvajeet Kumar**

GitHub: https://github.com/vishva-jeet/British-Airways-booking-prediction
