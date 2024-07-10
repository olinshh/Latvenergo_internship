import xgboost as xgb
import sklearn as sk
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('personal_project/euro2024_players.csv') # reading the csv file


positions = data["Position"]
encoder = OneHotEncoder()
encoded_positions_df = pd.DataFrame()


positions_list = ["Goalkeeper", "Centre-Back", "Left-Back", "Right-Back", "Defensive Midfield", "Central Midfield", "Attacking Midfield", "Left Winger", "Right Winger", "Second Striker", "Centre-Forward"]
for position in positions_list:
    encoded_positions_df[position] = (data['Position'] == position).astype(int)

#print(encoded_positions_df)
concat_list = [encoded_positions_df, data[["Goals", "Caps", "Foot", "Height", "Age"]]]


X = pd.concat(concat_list, axis = "columns")
#i need to convert the foot column to 0 and 1:
X["Foot"] = X["Foot"].replace({"both":3, "left":2, "right": 1, "-": 0})
y = data["MarketValue"]

column_names = X.columns

X["Goals/Caps"] = X["Goals"]/ X["Caps"]
X["Goals/Caps"].fillna(0, inplace = True)
X["Goals/Age"] = X["Goals"] / X["Age"]
X["Caps/Age"] = X["Caps"] / X["Age"] # there is no possibility of division by zero in these tw new columns so i dont have to fill na values with zeroes.

new_important_feature_df = X[["Age", "Caps", "Goals", "Height", "Goals/Caps", "Goals/Age", "Caps/Age"]] # trying only the most important features, heavily leading the scores of importance 

X_train, X_test, y_train, y_test = train_test_split(new_important_feature_df, y, test_size = 0.2, random_state = 42, shuffle = True)

xgb_model = xgb.XGBRegressor(n_estimators = 1000, max_depth = 7, eta = 0.1, subsample = 0.7, colsample_bytree = 0.8)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)     # first attempt at model
mae = mean_absolute_error(y_test, y_pred)

print("Mean squared error: ", mse)
print("Mean absolute error: ", mae)