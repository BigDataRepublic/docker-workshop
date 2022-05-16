from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd

app = FastAPI()

lgbm_model = joblib.load('model/lgbm_model.joblib')
enc = joblib.load('model/encoder.joblib')

@app.get("/")
def welcome_msg():
    return{"message":"Hello World!"}

@app.post("/predict")
def return_prediction(query: str):
    query = pd.read_json(query)
    query = enc.transform(query)
    return{"message":"Success", "Prediction":lgbm_model.predict(query)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
