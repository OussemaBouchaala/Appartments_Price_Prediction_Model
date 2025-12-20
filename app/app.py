from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from config import MODEL_RF_OPTIMIZED, SCALER, MODEL_COLUMNS
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Tunisia Real Estate Price Prediction API",
    description="API for predicting real estate prices in Tunisia",
    version="1.0.0"
)

# Load model artifacts at startup
try:
    model = joblib.load(MODEL_RF_OPTIMIZED)
    scaler = joblib.load(SCALER)
    model_columns = joblib.load(MODEL_COLUMNS)
    
    # Extract available cities
    city_cols = [c for c in model_columns if c.startswith('city_')]
    available_cities = [c.replace('city_', '') for c in city_cols]
    
except FileNotFoundError as e:
    raise RuntimeError(f"Model artifacts not found: {e}")


# Define request/response models
class PropertyInput(BaseModel):
    """Input schema for property prediction"""
    superficie: float = Field(..., ge=20, le=1000, description="Surface area in m²")
    chambres: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bains: int = Field(..., ge=1, le=5, description="Number of bathrooms")
    city: str = Field(..., description="City name")
    
    class Config:
        schema_extra = {
            "example": {
                "superficie": 120,
                "chambres": 3,
                "bains": 2,
                "city": "Tunis"
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction response"""
    estimated_price: float = Field(..., description="Predicted price in TND")
    property_details: dict = Field(..., description="Input property details")
    

# API Endpoints
@app.get("/", tags=["General"])
def root():
    """Welcome endpoint"""
    return {
        "message": "Tunisia Real Estate Price Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["General"])
def health_check():
    """Check if the API and model are loaded correctly"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.get("/cities", tags=["Info"])
def get_available_cities() -> List[str]:
    """Get list of available cities for prediction"""
    return sorted(available_cities)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_price(property_data: PropertyInput):
    """
    Predict real estate price based on property features
    
    - **superficie**: Surface area in m² (20-1000)
    - **chambres**: Number of bedrooms (1-10)
    - **bains**: Number of bathrooms (1-5)
    - **city**: City name (use /cities endpoint to get available cities)
    """
    
    # Validate city
    if property_data.city not in available_cities:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid city. Available cities: {', '.join(sorted(available_cities))}"
        )
    
    try:
        # 1. Prepare input dictionary with all model columns set to 0
        input_data = {col: 0 for col in model_columns}
        
        # 2. Scale numerical features
        df_numerical = pd.DataFrame(
            [[property_data.superficie, property_data.chambres, property_data.bains]], 
            columns=['superficie', 'chambres', 'bains']
        )
        scaled_values = scaler.transform(df_numerical)
        
        # Update input with scaled values
        input_data['superficie'] = scaled_values[0][0]
        input_data['chambres'] = scaled_values[0][1]
        input_data['bains'] = scaled_values[0][2]
        
        # 3. One-hot encode city
        city_col_name = f"city_{property_data.city}"
        if city_col_name in input_data:
            input_data[city_col_name] = 1
        
        # 4. Convert to DataFrame and predict
        input_df = pd.DataFrame([input_data])
        log_prediction = model.predict(input_df)[0]
        
        # 5. Transform back from log scale
        estimated_price = np.expm1(log_prediction)
        
        return PredictionResponse(
            estimated_price=round(estimated_price, 2),
            property_details={
                "superficie": property_data.superficie,
                "chambres": property_data.chambres,
                "bains": property_data.bains,
                "city": property_data.city
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(properties: List[PropertyInput]):
    """
    Predict prices for multiple properties at once
    """
    results = []
    
    for prop in properties:
        try:
            prediction = predict_price(prop)
            results.append({
                "success": True,
                "data": prediction
            })
        except HTTPException as e:
            results.append({
                "success": False,
                "error": e.detail,
                "property": prop.dict()
            })
    
    return {"predictions": results, "total": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
