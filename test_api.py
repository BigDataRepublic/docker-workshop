import requests
import os

import pandas as pd

data = pd.read_csv(os.getcwd() + "/data/csgo_round_snapshots.csv")

data = data.drop("round_winner", axis=1)

row = data.loc[2]

r = requests.post("http://0.0.0.0:8010/predict", json={"data": row.to_json()})

print(r.text)
