import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI

app = FastAPI()

lgbm_model = joblib.load("model/lgbm_model.joblib")
enc = joblib.load("model/encoder.joblib")


@app.get("/")
def welcome_msg():
    return {"message": "Hello World!"}


@app.post("/predict")
def return_prediction(query: str):
    query = pd.read_json(query)
    query = enc.transform(query)
    return {
        "status_code": 200,
        "message": "Success",
        "prediction": lgbm_model.predict(query),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
