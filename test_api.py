import os

import pandas as pd

data = pd.read_csv(os.getcwd() + "/data/test_set.csv")

random_row = data.loc[1000]
