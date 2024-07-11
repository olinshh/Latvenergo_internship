import sklearn as sk
import xgboost as xgb
import numpy as np
from xgboost import XGBClassifier as xgbclassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

data = pd.read_csv('personal_project/diabetes_project/diabetes.csv')

num_rows = data.shape[0]
print("the number of rows is: ", num_rows)

y = data["Outcome"]
X = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]

#change the insulin column to 0 and 1 meaning, if they have any value then put 1

X["Insulin"] = np.where(X["Insulin"] > 0, 1, 0) # since it has so much missing data, we resort to binary

# there are various columns that have missing data, therefore i should do to runs, one in which i 
# take the flawed data columns and use median values and another try when i dont use these columns at all
# theoretically, i could also do a run where i just take the data as is and see what happend
#remember to run feature importance tests !!!!!!!!


flawed_cols = [X["Glucose"], X["BloodPressure"], X["SkinThickness"], X["BMI"]]

for col in flawed_cols:
    col.replace(0, col.median(), inplace = True) # these columns have missing data

print("first 10 rows of dataframe post conversion of values", X[:10])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42, shuffle = True)

xgb_model = xgbclassifier(n_estimators = 1000, max_depth = 7, eta = 0.1, subsample = 0.7, colsample_bytree = 0.8)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy is: ", accuracy)

y_comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
x_test_df = pd.DataFrame(X_test)
#print("first ten of y table: ", y_comparison_df[:10])

#print("first ten of x test table: ", x_test_df[:10])

combined_df = pd.concat([y_comparison_df, x_test_df], axis = "columns")

print("combined data frame: ")
print(combined_df[:])

feature_importance = xgb_model.feature_importances_

feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
feature_importance_df.sort_values(by = "Importance", ascending = False, inplace = True)
print(feature_importance_df)

zero_count_insulin = X["Insulin"].value_counts()
print("Number of zeros in insulin column: ", zero_count_insulin)