import os
import pickle

import matplotlib.pyplot as plt
import mlflow
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

from log_config import configure_logger

# Configuring logger
log_file = False
Console = True
Log_level = "DEBUG"
log = configure_logger(log_file, Console, Log_level)


# Testing the package pack_install
def package_install_status():
    required_package_list = ["os", "pickle", "matplotlib", "numpy", "pandas", "scipy", "sklearn"]
    Package_status = pk.package_installation_status(required_package_list)
    mlflow.log_param("train_Package_status", Package_status)


# Function to write unit test cases
def subtract(x, y):
    """Subtract Function"""
    return x - y


sub = subtract(2, 3)

# Paths
housing_path = "./data/process"
artifacts_path = "./artifacts"


# Function to load CSV data
def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    mlflow.log_param("housing_path", housing_path)
    housing = pd.read_csv(csv_path)

    # Loading the CSV data
    # housing = load_housing_data(housing_path)
    log.debug("housing data is loaded for processing")

    mlflow.log_param("shape of housing data is", housing.shape)
    mlflow.log_param("Column names in housing data", housing.columns.values)

    # Bucketing income in income categories
    housing["income_cat"] = pd.cut(
        housing["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5]
    )

    return housing


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def train_and_test_split(housing):
    log.debug("Test/Train of data starts")

    # We are using two different types of train test split, Stratified Shuffle Split and Random Split

    # 1) Stratified Shuffle Split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    log.debug("Stratified Shuffle Split is complited")

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        start_train_set = housing.loc[train_index]
        start_test_set = housing.loc[test_index]

    mlflow.log_param("shape of start_train_set data is", start_train_set.shape)
    mlflow.log_param("Column names in start_train_set data", start_train_set.columns.values)

    mlflow.log_param("shape of start_test_set data is", start_test_set.shape)
    mlflow.log_param("Column names in start_test_set data", start_test_set.columns.values)

    # 2) Random Split
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    log.debug("Random Split is complited")

    mlflow.log_param("shape of train_set data is", train_set.shape)
    mlflow.log_param("Column names in train_set data", train_set.columns.values)

    mlflow.log_param("shape of test_set data is", test_set.shape)
    mlflow.log_param("Column names in test_set data", test_set.columns.values)

    return housing, start_train_set, start_test_set, train_set, test_set


def EDA(housing, start_train_set, start_test_set, train_set, test_set):
    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(start_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    log.debug("income_cat_proportions method is called")

    compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
    mlflow.log_param("random split error", compare_props["Rand. %error"])

    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )
    mlflow.log_param("Stratified Shuffle Split error", compare_props["Strat. %error"])

    """After Comparison of errors, %error of Stratified Shuffle Split < %error of random Split
    Therefor, we re progressing with train test split given by Stratified Shuffle Split"""

    # Dropping income_cat column from train and test dataset
    for set_ in (start_train_set, start_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    log.debug("Creating train set")
    train = start_train_set.copy()

    return train


def process_train_data(train, start_train_set):
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

    mlflow.log_param("Shape of train_prepared set", train_prepared.shape)
    mlflow.log_param("Column names of train_prepared set", train_prepared.columns.values)

    return train_prepared, train_labels


def model_training(train_prepared, train_labels):
    # We are using three different type of regression algorithams to train the model
    # 1) LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(train_prepared, train_labels)
    log.debug("LinearRegression model is trained")
    mlflow.sklearn.log_model(lin_reg, "Linear regression model")

    # 2) DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(train_prepared, train_labels)
    log.debug("DecisionTreeRegressor model is trained")
    mlflow.sklearn.log_model(tree_reg, "Decision tree model")

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
    mlflow.log_param("rnd_best_params", best_params)

    cvres = rnd_search.cv_results_

    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print("rnd_search_cv", np.sqrt(-mean_score), params)

    feature_importance_scores = rnd_search.best_estimator_.feature_importances_
    # mlflow.log_metric(key="rnd_feature_importance_scores", value=feature_importance_scores)

    mlflow.sklearn.log_model(rnd_search, "Random forest random search model")

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
    mlflow.log_param("grid_best_params", best_params)

    cvres = grid_search.cv_results_

    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print("grid_search_cv", np.sqrt(-mean_score), params)

    feature_importance_scores = grid_search.best_estimator_.feature_importances_
    mlflow.log_param("grid_feature_importance_scores", feature_importance_scores)

    Features_with_Scores = sorted(
        zip(feature_importance_scores, train_prepared.columns), reverse=True
    )
    # mlflow.log_metric(key="grid_Features_with_Scores", value=Features_with_Scores)

    final_forest_reg_model = grid_search.best_estimator_
    mlflow.log_param("final_forest_reg_model", final_forest_reg_model)

    mlflow.sklearn.log_model(grid_search, "Random forest grid search model")

    return final_forest_reg_model, lin_reg, tree_reg, forest_reg


def process_test_data(start_test_set):
    # Preparing testing data
    log.debug("Creating Test data")

    test = start_test_set.drop("median_house_value", axis=1)

    test_lables = start_test_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    test_num = test.drop("ocean_proximity", axis=1)
    imputer.fit(test_num)  # Find the Meadian of respective columns
    X = imputer.transform(test_num)
    test_prepared = pd.DataFrame(X, columns=test_num.columns, index=test.index)
    test_prepared["rooms_per_household"] = (
        test_prepared["total_rooms"] / test_prepared["households"]
    )
    test_prepared["bedrooms_per_room"] = (
        test_prepared["total_bedrooms"] / test_prepared["total_rooms"]
    )
    test_prepared["population_per_household"] = (
        test_prepared["population"] / test_prepared["households"]
    )

    X_test_cat = test[["ocean_proximity"]]
    test_prepared = test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    mlflow.log_param("Shape of test_prepared set", test_prepared.shape)
    mlflow.log_param("Column names of test_prepared set", test_prepared.columns.values)

    return test_prepared, test_lables


def model_object_pkl(
    artifacts_path, lin_reg, tree_reg, forest_reg, test_lables, final_forest_reg_model
):
    # Saving Model Object in pickle folder
    mlflow.log_param("artifacts_path", artifacts_path)
    file = open(os.path.join(artifacts_path, "model_object.pkl"), "wb")
    mlflow.log_artifact(artifacts_path, "model_object.pkl")
    pickle.dump(lin_reg, file)
    pickle.dump(tree_reg, file)
    pickle.dump(forest_reg, file)
    pickle.dump(test_lables, file)
    pickle.dump(final_forest_reg_model, file)
    file.close()
    log.debug("Model objects in pickle file are dumped")


def train_main():

    # Calling all the methods
    package_install_status()
    housing = load_housing_data(housing_path)
    housing, start_train_set, start_test_set, train_set, test_set = train_and_test_split(housing)
    train = EDA(housing, start_train_set, start_test_set, train_set, test_set)
    train_prepared, train_labels = process_train_data(train, start_train_set)
    final_forest_reg_model, lin_reg, tree_reg, forest_reg = model_training(
        train_prepared, train_labels
    )
    test_prepared, test_lables = process_test_data(start_test_set)
    model_object_pkl(
        artifacts_path, lin_reg, tree_reg, forest_reg, test_lables, final_forest_reg_model
    )

    # Writing the test data to csv file
    test_prepared.to_csv(os.path.join(housing_path, "test.csv"), index=False)
    log.debug("Test data is prepared")
    mlflow.log_param("Shape of test data", test_prepared.shape)
    mlflow.log_param("columns in Test data", test_prepared.columns.values)


if __name__ == "__main__":

    train_main()

    printout()
