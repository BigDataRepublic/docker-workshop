import os
import pickle

import lightgbm as lgb
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv(os.getcwd() + "/data/csgo_round_snapshots.csv")

# Map the team name to integer target variable
team_dict = {"CT": 0, "T": 1}
data["round_winner"] = data["round_winner"].map(team_dict)

X = data.drop("round_winner", axis=1)
y = data["round_winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

X_train.to_csv(os.getcwd() + "/data/train_set.csv")
X_test.to_csv(os.getcwd() + "/data/test_set.csv")

# one-hot encode categorical values
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(X_train[['map']]).toarray())
X_train = X_train.join(enc_df)
X_train = X_train.drop("map", axis=1)

enc_df = pd.DataFrame(enc.transform(X_test[['map']]).toarray())
X_test = X_test.join(enc_df)
X_test = X_test.drop("map", axis=1)

lgbm_model = lgb.LGBMClassifier(n_estimators=25000)
lgbm_model.fit(X_train, y_train)

print(lgbm_model.score(X_test, y_test))

p = pickle.dumps(lgbm_model)

dump(lgbm_model, "model/lgbm_model.joblib", compress=3)
dump(enc, "model/encoder.joblib")
