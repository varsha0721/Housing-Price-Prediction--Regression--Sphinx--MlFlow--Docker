import os
import pickle

import mlflow
import numpy as np
import pandas as pd
from logging_tree import printout
from pack_install import find_installed_package as pk
from sklearn.metrics import mean_absolute_error, mean_squared_error

from log_config import configure_logger

# Paths
housing_path = "./data/process"
output_path = "./data/output"
artifacts_path = "./artifacts"

# Configuring logger
log_file = False
Console = True
Log_level = "DEBUG"
log = configure_logger(log_file, Console, Log_level)


# Testing the package pack_install
def package_install_status():
    required_package_list = ["os", "pickle", "numpy", "pandas", "sklearn"]
    Package_status = pk.package_installation_status(required_package_list)
    log.debug(Package_status)
    mlflow.log_param("score_Package_status", Package_status)


# Function to write unit test cases
def divide(x, y):
    """Divide Function"""
    if y == 0:
        raise ValueError("Can not divide by zero!")
    return x / y


div = divide(15, 5)


def load_test_data():
    test = pd.read_csv(os.path.join(housing_path, "test.csv"))
    log.debug("Test data is prepared")
    mlflow.log_param("Shape of test data", test.shape)
    mlflow.log_param("columns in Test data", test.columns.values)

    return test


def load_model_objects(artifacts_path):
    # Loading Model Objects form pickle file
    mlflow.log_param("artifacts_path", artifacts_path)
    file = open(os.path.join(artifacts_path, "model_object.pkl"), "rb")
    mlflow.log_artifact(artifacts_path, "model_object.pkl")
    lin_reg = pickle.load(file)
    tree_reg = pickle.load(file)
    forest_reg = pickle.load(file)
    test_labels = pickle.load(file)
    final_forest_reg_model = pickle.load(file)
    file.close()
    log.debug("model objects from pickle file are loaded")

    return lin_reg, tree_reg, forest_reg, test_labels, final_forest_reg_model


def model_testing(test, lin_reg, tree_reg, forest_reg, test_labels, final_forest_reg_model):
    # Prediction using LinearRegression Model
    lin_reg_housing_predictions = lin_reg.predict(test)

    lin_mae = mean_absolute_error(test_labels, lin_reg_housing_predictions)
    lin_mse = mean_squared_error(test_labels, lin_reg_housing_predictions)
    lin_rmse = np.sqrt(lin_mse)

    log.debug("Model performence measures for LinearRegression")
    mlflow.log_metric(key="LinearRegression_MAE", value=lin_mae)
    mlflow.log_metric(key="LinearRegression_MSE", value=lin_mse)
    mlflow.log_metric(key="LinearRegression_RMSE", value=lin_rmse)

    # Prediction using DecisionTreeRegressor Model
    tree_reg_housing_predictions = tree_reg.predict(test)

    tree_mae = mean_absolute_error(test_labels, tree_reg_housing_predictions)
    tree_mse = mean_squared_error(test_labels, tree_reg_housing_predictions)
    tree_rmse = np.sqrt(tree_mse)

    log.debug("Model performence measures for DecisionTreeRegressor")
    mlflow.log_metric(key="DecisionTreeRegressor_MAE", value=tree_mae)
    mlflow.log_metric(key="DecisionTreeRegressor_MSE", value=tree_mse)
    mlflow.log_metric(key="DecisionTreeRegressor_RMSE", value=tree_rmse)

    # Prediction using RandomForestRegressor Model
    final_forest_reg_housing_predictions = final_forest_reg_model.predict(test)

    Forest_mae = mean_absolute_error(test_labels, final_forest_reg_housing_predictions)
    forest_mse = mean_squared_error(test_labels, final_forest_reg_housing_predictions)
    forest_rmse = np.sqrt(forest_mse)

    log.debug("Model performence measures for RandomForestRegressor")
    mlflow.log_metric(key="RandomForestRegressor_MAE", value=Forest_mae)
    mlflow.log_metric(key="RandomForestRegressor_MSE", value=forest_mse)
    mlflow.log_metric(key="RandomForestRegressor_RMSE", value=forest_rmse)

    # As the errors are less for RandomForestRegressor we are using this model's predictions in output file.
    return final_forest_reg_housing_predictions


def prediction_output(Prediction_output, output_path):
    # Writing the file with predicted house price
    mlflow.log_param("output_path", output_path)
    Prediction_output.to_csv(
        os.path.join(
            output_path,
            "Prediction_output_{}.csv".format(pd.datetime.now().strftime("%Y-%m-%d %H%M%S")),
        ),
        index=False,
    )
    log.debug("Prediction file is created")


def score_main():
    # Calling Functions
    test = load_test_data()
    lin_reg, tree_reg, forest_reg, test_labels, final_forest_reg_model = load_model_objects(
        artifacts_path
    )
    final_forest_reg_housing_predictions = model_testing(
        test, lin_reg, tree_reg, forest_reg, test_labels, final_forest_reg_model
    )

    Prediction_output = test.copy()
    Prediction_output["Predicted House Price"] = final_forest_reg_housing_predictions

    prediction_output(Prediction_output, output_path)


if __name__ == "__main__":
    score_main()
    printout()
