import sklearn as sk
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from sklearn.ensemble import RandomForestClassifier



#testing pandas, matplotlib:

data = pd.read_csv('personal_project/euro2024_players.csv')

# plt.scatter(data.Age, data.Goals)

# plt.title("age vs goals among euro 2024 players")
# plt.xlabel("age")                                       #age vs goals scatterplot
# plt.ylabel("goals scored")
# plt.savefig("personal_project/age_vs_goals.png")
# plt.show()


# left = data.loc[data["Foot"] == "left"]
# right = data.loc[data["Foot"] == "right"]
# labels = ["Left", "Right"]                                          # left right foot pie chart
# colors = ["red", "blue"]
# plt.pie([len(left), len(right)], labels = labels, colors = colors, autopct= '%1.1f%%')
# plt.savefig("personal_project/left_vs_right_pie.png")
# plt.show()


andrich = data.loc[data["Name"] == "Robert Andrich"]
pavlovic = data.loc[data["Name"] == "Aleksandar Pavlovic"]              #loc is amethod to retrieve rows from data set

print(andrich, "\n\n\n", pavlovic)

# what to do next? :
#describe head, tail, what means?
# how to train the data ?
# should i rename all of the "left", "Ŗight occurances in the table to 0 and 1?"
# i want to make a fnction that describe a potential player's cost based on the following input: position, goals, caps (international games played), foot, height, age
# i will later remove the negligible variable to improve th accuracy of the model = FEATURE IMPORTANCE
# i need to split the data i have into training and testing sets
#random forest regressor? xg boost?


# testing numpy:

goals_median = np.median(data.Goals)
goals_average = np.mean(data.Goals)
goals_min = np.min(data.Goals)
goals_max = np.max(data.Goals)

print("goals median: ", goals_median)
print("goals average: ", goals_average)             # some stats about goals
print("goals min: ", goals_min)
print("goals max: ", goals_max)

sorted_heights = np.sort(data.Height)
median_height = np.median(sorted_heights)         #height stats
print("heights sorted: ", sorted_heights)
print("median height: ", median_height)

# plt.boxplot(data.Height)
# plt.title("Boxplot of player heights")            # boxplot of heights
# plt.ylabel("height in cm")
# plt.savefig("personal_project/boxplot_heights2.png", format = 'png')
# plt.show()

# fig = plt.figure()
# threeD = fig.add_subplot(111, projection = '3d')
# threeD.scatter(data.Caps, data.Goals, data.MarketValue)         # making a 3d scatter plot
# threeD.set_xlabel("caps")
# threeD.set_ylabel("goals scored")
# threeD.set_zlabel("market value")

# plt.savefig("personal_project/3d_scatter.png", format = 'png')
# plt.show()


# plt.plot(data.Goals, data.MarketValue, 'r.')
# m, b = np.polyfit(data.Goals, data.MarketValue, 1)            # plotting regression line of goals vs market value
# plt.plot(data.Goals, m*data.Goals + b)
# plt.xlabel("goals")
# plt.ylabel("market value")
# plt.savefig("bad_regression_goals_vs_price.png")


# correlation_goals_value = np.corrcoef(data.Goals, data.MarketValue)
# print("The correlation between goals and market value is: ", correlation_goals_value[0][1])

# plt.show()


#TESTING SKLEARN:


positions = data["Position"]
encoder = OneHotEncoder()
encoded_positions_df = pd.DataFrame()
positions_list = ["Goalkeeper", "Centre-Back", "Left-Back", "Right-Back", "Defensive Midfield", "Central Midfield", "Attacking Midfield", "Left Winger", "Right Winger", "Second Striker", "Centre-Forward",]
for position in positions_list:
    encoded_positions_df[position] = (data['Position'] == position).astype(int)

#print(encoded_positions_df)
concat_list = [encoded_positions_df, data[["Goals", "Caps", "Foot", "Height", "Age"]]]

X = pd.concat(concat_list, axis = "columns")
#i need to convert the foot column to 0 and 1:
X["Foot"] = X["Foot"].replace({"both":3, "left":2, "right": 1, "-": 0})


print("The first ten rows of super data frame", X.iloc[:10])
column_names = X.columns
print("the column names of super data frame are: ", column_names)      # i need to give the columns that hold positions less weightage

y = data["MarketValue"]
print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X[2:], y[2:], test_size = 0.2, random_state = 42, shuffle = True)
print("The x train data is:", X_train)

# #later i will need to reevaluate the features that I have selected based on how much noise or signal they give
# #i need to choose an appropriate model: e.g. decision tree regressor, random forest regressor, xg boost regressor
# # need to train the model
# #hyperparameter tuning
# #evalueate the model
# # deploy the model to make predictions on new player data


rf_regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)

# flag = False

# for i in X[:20]:
#     if(type(X[i]) != int):
#         print("bug is:", X[i], "/n/n")
#         print("bug type: ", type(X[i]))
#         print("row of occurance: ", i)
#         flag = True
#     if (flag):
#         break



rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)     # first attempt at model
mae = mean_absolute_error(y_test, y_pred)

print("Mean absolute error1: ", mae)
print("Mean squared error1: ", mse)

# future plans: FEATURE IMPORTANCE, TRYING ANOTHER MODEL ON SAME DATA, TRYING OTHER DATA
# ALSO: ALLOWING USER TO MAKE CUSTOM PLAYER BASED ON MADEUP STATISTICS THAT SPITS OUT A PREDICTED MARKET VALUE

# trying to evaluate feature importances:
importance = rf_regressor.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
feature_importance_df = feature_importance_df.sort_values("Importance", ascending = False)

print(feature_importance_df)

new_important_feature_df = X[["Age", "Caps", "Goals", "Height"]] # trying only the most important features, heavily leading the scores of importance 

X_train2, X_test2, y_train2, y_test2 = train_test_split(new_important_feature_df, y, test_size = 0.2, random_state = 42, shuffle = True)

rf_regressor2 = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf_regressor2.fit(X_train2, y_train2)
y_pred2 = rf_regressor2.predict(X_test2)


mse2 = mean_squared_error(y_test2, y_pred2)
mae2 = mean_absolute_error(y_test2, y_pred2)

print("Mean absolute error2: ", mae2)
print("Mean squared error2: ", mse2)


# add new features: goals/caps, goals/age, caps/age

X["Goals/Caps"] = X["Goals"]/ X["Caps"]
X["Goals/Caps"].fillna(0, inplace = True)
print("Goals/caps column testing: ", X["Goals/Caps"][:10])       # adding, testing new features
print("Goals:", X["Goals"][:10])
print("Caps:", X["Caps"][:10])

X["Goals/Age"] = X["Goals"] / X["Age"]
print("first 10 rows of goals/age: ", X["Goals/Age"][:10])
X["Caps/Age"] = X["Caps"] / X["Age"] # there is no possibility of division by zero in these tw new columns so i dont have to fill na values with zeroes.
print("first 10 rows of caps/age: ", X["Caps/Age"][:10])

# trying random forest regressor with new features last time:

new_important_feature_df3 = X[["Age", "Caps", "Goals", "Height", "Goals/Caps", "Goals/Age", "Caps/Age"]] # trying only the most important features, heavily leading the scores of importance 

X_train3, X_test3, y_train3, y_test3 = train_test_split(new_important_feature_df3, y, test_size = 0.2, random_state = 42, shuffle = True)

rf_regressor3 = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf_regressor3.fit(X_train3, y_train3)
y_pred3 = rf_regressor3.predict(X_test3)


mse3 = mean_squared_error(y_test3, y_pred3)
mae3 = mean_absolute_error(y_test3, y_pred3)

print("Mean absolute error3: ", mae3)
print("Mean squared error3: ", mse3)

importance3 = rf_regressor3.feature_importances_
feature_names3 = new_important_feature_df3.columns

feature_importance_df3 = pd.DataFrame({"Feature": feature_names3, "Importance": importance3})
feature_importance_df3 = feature_importance_df3.sort_values("Importance", ascending = False)

print(feature_importance_df3)

#next up: make a regression model using xg boost
