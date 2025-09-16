# Step 1: Import Libraries
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np

# Step 2: Load the Trained Model
pipe = pickle.load(open(r"C:\Users\Kumarjit Brahma\LinearRegressionModel.pkl", "rb"))

# Step 3: Initialize FastAPI
app = FastAPI(title="Car Price Prediction API", 
              description="Predict car prices using a trained Linear Regression model with preprocessing pipeline.",
              version="1.0")

# Step 4: Define Input Schema
class CarFeatures(BaseModel):
    name: str
    company: str
    year: int
    kms_driven: int
    fuel_type: str

# Step 5: Create Prediction Endpoint
@app.post("/predict")
def predict(car: CarFeatures):
    """
    Predict the price of a car given its features.
    
    Parameters:
        car (CarFeatures): Input car details.
        
    Returns:
        dict: Predicted car price.
    """
    # Convert input to DataFrame
    df = pd.DataFrame([car.dict()])
    
    # Apply feature engineering consistent with training
    df['car_age'] = 2025 - df['year']            # Replace 2025 with current year if needed
    df['log_kms'] = np.log1p(df['kms_driven'])
    
    # Select features used during training
    X = df[['name', 'company', 'car_age', 'log_kms', 'fuel_type']]
    
    # Make prediction using the pipeline
    prediction = pipe.predict(X)[0]
    
    # Return prediction
    return {"predicted_price": round(prediction, 2)}

