import os

import pandas as pd
import requests

data = pd.read_csv(os.getcwd() + "/data/test_set.csv")

random_row = data.loc[1000]

r = requests.post("http://0.0.0.0:8000/predict", json={"data": random_row.to_json()})

print(r.text)
