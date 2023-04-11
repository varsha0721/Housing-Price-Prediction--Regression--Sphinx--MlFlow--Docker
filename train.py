import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logging_tree import printout
from pack_install import find_installed_package as pk
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

import ingest_data
from log_config import configure_logger

# Configuring logger
log_file = False
Console = True
Log_level = "DEBUG"
log = configure_logger(log_file, Console, Log_level)

# Testing the package pack_install
required_package_list = ["os", "pickle", "matplotlib", "numpy", "pandas", "scipy", "sklearn"]
Package_status = pk.package_installation_status(required_package_list)
log.debug(Package_status)

# Function to write unit test cases
def subtract(x, y):
    """Subtract Function"""
    return x - y


sub = subtract(2, 3)

# Using the same path from ingest_data.py
housing_path = ingest_data.PROCESS_PATH
artifacts_path = ingest_data.ARTIFACTS_PATH

# Function to load CSV data
def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# Loading the CSV data
housing = load_housing_data(housing_path)
log.debug("housing data is loaded for processing")
log.debug("shape of data is", housing.shape)
log.debug("Column names in housing data", housing.columns.values)

# Bucketing income in income categories
housing["income_cat"] = pd.cut(
    housing["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5]
)

log.debug("Test/Train of data starts")
# We are using two different types of train test split, Stratified Shuffle Split and Random Split

# 1) Stratified Shuffle Split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
log.debug("Stratified Shuffle Split is complited")

for train_index, test_index in split.split(housing, housing["income_cat"]):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]

# 2) Random Split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
log.debug("Random Split is complited")


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


compare_props = pd.DataFrame(
    {
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(start_test_set),
        "Random": income_cat_proportions(test_set),
    }
).sort_index()
log.debug("income_cat_proportions method is called")

compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
log.debug("random split %error", compare_props["Rand. %error"])

compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
log.debug('Stratified Shuffle Split %error"', compare_props["Strat. %error"])

"""After Comparison of errors, %error of Stratified Shuffle Split < %error of random Split
Therefor, we re progressing with train test split given by Stratified Shuffle Split"""

# Dropping income_cat column from train and test dataset
for set_ in (start_train_set, start_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

log.debug("Creating train set")
train = start_train_set.copy()
log.debug("Shape of train set", train.shape)
log.debug("Column names of train set", train.columns.values)

# Plot to see the relation between x and y
train.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()

"""From plot we can say that x and y are (-ve)ly correlated"""

corr_matrix = train.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

"""From above corrilation we can conclude that median_house_value has no linear correlation with any of other features
 as all the correlation values are nearly equal to 0"""

train["rooms_per_household"] = train["total_rooms"] / train["households"]
train["bedrooms_per_room"] = train["total_bedrooms"] / train["total_rooms"]
train["population_per_household"] = train["population"] / train["households"]


train = start_train_set.drop("median_house_value", axis=1)  # drop labels for training set

train_labels = start_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")

train_num = train.drop("ocean_proximity", axis=1)

imputer.fit(train_num)  # Find the Meadian of respective columns
X = imputer.transform(train_num)  # Repalce all null valus with Median of the respective columns

train_tr = pd.DataFrame(X, columns=train_num.columns, index=train.index)
train_tr["rooms_per_household"] = train_tr["total_rooms"] / train_tr["households"]
train_tr["bedrooms_per_room"] = train_tr["total_bedrooms"] / train_tr["total_rooms"]
train_tr["population_per_household"] = train_tr["population"] / train_tr["households"]

housing_cat = train[["ocean_proximity"]]

# Convert text valus to numbers
# Final Training dataset prepared
train_prepared = train_tr.join(pd.get_dummies(housing_cat, drop_first=True))
log.debug("Training data is prepared")

# We are using three different type of regression algorithams to train the model
# 1) LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_prepared, train_labels)
log.debug("LinearRegression model is trained")

# 2) DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(train_prepared, train_labels)
log.debug("DecisionTreeRegressor model is trained")

"""To find the best set of perameters for RandomForestRegressor we are using two methods
   RandomForestRegressor and GridSearchCV"""


# 3) RandomForestRegressor
param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_distribs,
    n_iter=10,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
)
rnd_search.fit(train_prepared, train_labels)
log.debug("RandomForestRegressor model is trained with RandomizedSearchCV parameters")

best_params = rnd_search.best_params_

cvres = rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print("rnd_search_cv", np.sqrt(-mean_score), params)

feature_importance_scores = rnd_search.best_estimator_.feature_importances_


param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(
    forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True,
)

grid_search.fit(train_prepared, train_labels)
log.debug("RandomForestRegressor model is trained with GridSearchCV parameters")

best_params = grid_search.best_params_

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print("grid_search_cv", np.sqrt(-mean_score), params)

feature_importance_scores = grid_search.best_estimator_.feature_importances_

Features_with_Scores = sorted(zip(feature_importance_scores, train_prepared.columns), reverse=True)
# print(Features_with_Scores)

final_forest_reg_model = grid_search.best_estimator_

# Preparing testing data
log.debug("Creating Test data")
test = start_test_set.drop("median_house_value", axis=1)

test_lables = start_test_set["median_house_value"].copy()

test_num = test.drop("ocean_proximity", axis=1)
test_prepared = imputer.transform(test_num)
test_prepared = pd.DataFrame(test_prepared, columns=test_num.columns, index=test.index)
test_prepared["rooms_per_household"] = test_prepared["total_rooms"] / test_prepared["households"]
test_prepared["bedrooms_per_room"] = test_prepared["total_bedrooms"] / test_prepared["total_rooms"]
test_prepared["population_per_household"] = (
    test_prepared["population"] / test_prepared["households"]
)

X_test_cat = test[["ocean_proximity"]]
test_prepared = test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

# Saving Model Object in pickle folder
file = open(os.path.join(artifacts_path, "model_object.pkl"), "wb")
pickle.dump(lin_reg, file)
pickle.dump(tree_reg, file)
pickle.dump(forest_reg, file)
pickle.dump(test_lables, file)
pickle.dump(final_forest_reg_model, file)
file.close()
log.debug("Model objects in pickle file are dumped")

# Writing the test data to csv file
test_prepared.to_csv(os.path.join(housing_path, "test.csv"), index=False)
log.debug("Test data is prepared")
log.debug("Shape of test data", test_prepared.shape)
log.debug("columns in Test data", test_prepared.columns.values)

printout()
