from pydantic import BaseModel

class GREInput(BaseModel):
    gre_score: float
    toefl_score: float
    university_rating: float
    sop: float
    lor: float
    cgpa: float
    research: int

import joblib
import numpy as np
from fastapi import FastAPI
from tensorflow.keras.models import load_model

app = FastAPI()

# Load model
model = load_model("model.keras")
scaler = joblib.load("StandardScaler.pkl")

@app.get("/")
def home():
    return {"message": "GRE Predictor API Running"}

@app.post("/predict")
def predict(data: GREInput):
    features = np.array([[
        data.gre_score,
        data.toefl_score,
        data.university_rating,
        data.sop,
        data.lor,
        data.cgpa,
        data.research
    ]])

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    return {"prediction": float(prediction[0])}