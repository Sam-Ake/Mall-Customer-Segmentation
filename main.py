from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Customer Segmentation API",
    description="REST API for predicting customer segment based on Annual Income and Spending Score.",
    version="1.0.0"
)

# Load saved models
try:
    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model files: {e}")

# Mapping dictionary for business segment names
SEGMENT_LABELS = {
    0: 'Sensible (Moderate Income, Moderate Spending)',
    1: 'Careless (Low Income, High Spending)',
    2: 'Target / Stars (High Income, High Spending)',
    3: 'Careful (High Income, Low Spending)',
    4: 'Miser / Saver (Low Income, Low Spending)'
}

# Define input schema with validation
class CustomerData(BaseModel):
    annual_income: float = Field(..., ge=0, description="Annual Income in k$ (e.g., 15 to 150)", json_schema_extra={"example": 70.0})
    spending_score: float = Field(..., ge=1, le=100, description="Spending Score between 1 and 100", json_schema_extra={"example": 85.0})

@app.get("/")
def home():
    return {"message": "Customer Segmentation API is running. Go to /docs for Swagger UI."}

@app.post("/segment")
def predict_segment(data: CustomerData):
    try:
        # Prepare input array
        input_data = np.array([[data.annual_income, data.spending_score]])
        
        # Scale input data
        scaled_data = scaler.transform(input_data)
        
        # Predict cluster
        cluster_id = int(kmeans.predict(scaled_data)[0])
        segment_name = SEGMENT_LABELS.get(cluster_id, "Unknown Segment")
        
        return {
            "annual_income_k$": data.annual_income,
            "spending_score": data.spending_score,
            "cluster_id": cluster_id,
            "segment_name": segment_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))