import os

import joblib
import pandas as pd
from pydantic import BaseModel

# import uvicorn
from fastapi import FastAPI

lgbm_model = joblib.load(os.getcwd() + "/model/lgbm_model.joblib")
enc = joblib.load(os.getcwd() + "/model/encoder.joblib")
team_dict = {0: "CT", 1: "T"}


class Data(BaseModel):
    data: str


def welcome_message() -> str:
    """Welcome message to test the API."""
    return "Hello World!"


def return_prediction(data: Data) -> dict:
    """Return a prediction for a single example from the testset with our own ML model.

    Args:
        - query: integer with the index of the testset to generate a prediction for
    """
    team_dict = {0: "CT", 1: "T"}

    data = pd.read_json(data.data, typ="series").to_frame()

    data = data.T

    enc_df = pd.DataFrame(enc.transform(data[["map"]]).toarray())
    data = data.join(enc_df)
    data = data.drop("map", axis=1)
    data = data.astype("float")

    try:
        # Get prediction
        pred = lgbm_model.predict(data)

        predicted_proba = lgbm_model.predict_proba(data)[0][pred]

        pred_desc = team_dict[pred[0]]
        return pred, predicted_proba, pred_desc

    except:
        return None
