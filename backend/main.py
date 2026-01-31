from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import os

# ==============================
# Load ML model (safe path)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lung_cancer_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ lung_cancer_model.joblib file not found")

model = joblib.load(MODEL_PATH)

# ==============================
# FastAPI app
# ==============================
app = FastAPI(
    title="Lung Cancer Prediction API",
    version="1.0"
)

# ==============================
# CORS (Frontend connect)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # production এ specific domain দিবে
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Input Schema with STRICT validation
# ==============================
from pydantic import BaseModel, Field, field_validator

class LungCancerInput(BaseModel):
    gender: int = Field(..., ge=0, le=1, description="Gender (0: Male, 1: Female)")
    age: int = Field(..., gt=0, description="Age of the patient")
    smoking: int = Field(..., ge=1, le=2, description="Smoking (1: No, 2: Yes)")
    yellow_fingers: int = Field(..., ge=1, le=2, description="Yellow Fingers (1: No, 2: Yes)")
    anxiety: int = Field(..., ge=1, le=2, description="Anxiety (1: No, 2: Yes)")
    peer_pressure: int = Field(..., ge=1, le=2, description="Peer Pressure (1: No, 2: Yes)")
    chronic_disease: int = Field(..., ge=1, le=2, description="Chronic Disease (1: No, 2: Yes)")
    fatigue: int = Field(..., ge=1, le=2, description="Fatigue (1: No, 2: Yes)")
    allergy: int = Field(..., ge=1, le=2, description="Allergy (1: No, 2: Yes)")
    wheezing: int = Field(..., ge=1, le=2, description="Wheezing (1: No, 2: Yes)")
    alcohol_consuming: int = Field(..., ge=1, le=2, description="Alcohol Consuming (1: No, 2: Yes)")
    coughing: int = Field(..., ge=1, le=2, description="Coughing (1: No, 2: Yes)")
    shortness_of_breath: int = Field(..., ge=1, le=2, description="Shortness of Breath (1: No, 2: Yes)")
    swallowing_difficulty: int = Field(..., ge=1, le=2, description="Swallowing Difficulty (1: No, 2: Yes)")
    chest_pain: int = Field(..., ge=1, le=2, description="Chest Pain (1: No, 2: Yes)")

    # Blank / empty check
    @field_validator("*", mode="before")
    @classmethod
    def no_blank_value(cls, v):
        if v is None or v == "":
            raise ValueError("Field cannot be blank")
        return v

# ==============================
# Routes
# ==============================
@app.get("/")
def health():
    return {"status": "API running successfully"}

@app.post("/predict")
def predict(data: LungCancerInput):

    values = list(data.model_dump().values())


    features = np.array([values])

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "result": "Lung Cancer Detected" if prediction == 1 else "No Lung Cancer"
    }
