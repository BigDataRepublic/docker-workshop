import os

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

lgbm_model = joblib.load(os.getcwd() + "/model/lgbm_model.joblib")
enc = joblib.load(os.getcwd() + "/model/encoder.joblib")
team_dict = {0: "CT", 1: "T"}

description = "# CS:GO gamewinner prediction API"

app = FastAPI(description=description)


class Data(BaseModel):
    data: str


@app.get("/")
def welcome_message() -> dict:
    """Welcome message to test the API."""
    return {"message": "Hello World!"}


@app.post("/predict")
def return_prediction(payload: Data) -> dict:
    """Return a prediction for a single example from the testset with our own ML model.

    Args:
        - data: the jsonified row of data for a single example
    """
    try:
        # load data in correct format
        data = pd.read_json(payload.data, typ="series").to_frame()
        data = data.T

        # transform categorical data with one-hot encoder saved during training process
        enc_df = pd.DataFrame(enc.transform(data[["map"]]).toarray())
        data = data.join(enc_df)
        data = data.drop("map", axis=1)
        data = data.astype("float")

        # Get prediction
        pred = lgbm_model.predict(data)
        predicted_proba = lgbm_model.predict_proba(data)[0][pred]
        pred_desc = team_dict[pred[0]]

        return {
            "message": f"Lgbm model predicts '{pred_desc}' with a probability of {predicted_proba}",
        }

    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Something went wrong, please check your request. Error: {e}")
