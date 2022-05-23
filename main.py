import os

import joblib
import pandas as pd
from pydantic import BaseModel

# import uvicorn
from fastapi import FastAPI

lgbm_model = joblib.load(os.getcwd() + "/model/lgbm_model.joblib")
enc = joblib.load(os.getcwd() + "/model/encoder.joblib")
team_dict = {0: "CT", 1: "T"}

description = "# CS:GO gamewinner prediction API"

app = FastAPI(description=description, debug=True)


class Data(BaseModel):
    data: str


@app.get("/")
def welcome_message() -> dict:
    """Welcome message to test the API."""
    return {"message": "Hello World!"}


@app.get("/test_numbers")
def get_index_range() -> dict:
    """Get the index range of the saved test set to inform the user about the
    possible indices to generate a prediction for.
    """
    # data = pd.read_csv(os.getcwd() + "/data/test_set.csv")
    return {
        "status_code": 200,
        "message": f"Provide any index in the following range to the predict function: {data.index}",
    }


@app.post("/predict")
def return_prediction(data: Data) -> dict:
    """Return a prediction for a single example from the testset with our own ML model.

    Args:
        - query: integer with the index of the testset to generate a prediction for
    """
    # data = pd.read_csv(os.getcwd() + "/data/test_set.csv")
    team_dict = {0: "CT", 1: "T"}

    print("it's doing something")
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
        return {
            "status_code": 200,
            "message": f"Lgbm model predicts '{pred_desc}' with a probability of {predicted_proba}",
        }
    except:
        return {
            "status_code": 400,
            "message": f"Something went wrong, please check your request. ",
        }
