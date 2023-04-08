import os
import pickle

import numpy as np
import pandas as pd
from logging_tree import printout
from pack_install import find_installed_package as pk
from sklearn.metrics import mean_absolute_error, mean_squared_error

import ingest_data
from log_config import configure_logger

# Using the same path from ingest_data.py
housing_path = ingest_data.PROCESS_PATH
output_path = ingest_data.OUTPUT_PATH
artifacts_path = ingest_data.ARTIFACTS_PATH

# Configuring logger
log_file = False
Console = True
Log_level = "DEBUG"
log = configure_logger(log_file, Console, Log_level)

# Testing the package pack_install
required_package_list = ["os", "pickle", "numpy", "pandas", "sklearn"]
Package_status = pk.package_installation_status(required_package_list)
log.debug(Package_status)

# Function to write unit test cases
def divide(x, y):
    """Divide Function"""
    if y == 0:
        raise ValueError("Can not divide by zero!")
    return x / y


div = divide(15, 5)

test = pd.read_csv(os.path.join(housing_path, "test.csv"))
log.debug("Test data is prepared")
log.debug("Shape of test data", test.shape)
log.debug("columns in Test data", test.columns.values)

# Loading Model Objects form pickle file
file = open(os.path.join(artifacts_path, "model_object.pkl"), "rb")
lin_reg = pickle.load(file)
tree_reg = pickle.load(file)
forest_reg = pickle.load(file)
test_labels = pickle.load(file)
final_forest_reg_model = pickle.load(file)
file.close()
log.debug("model objects from pickle file are loaded")

# Prediction using LinearRegression Model
lin_reg_housing_predictions = lin_reg.predict(test)

lin_mae = mean_absolute_error(test_labels, lin_reg_housing_predictions)
lin_mse = mean_squared_error(test_labels, lin_reg_housing_predictions)
lin_rmse = np.sqrt(lin_mse)

log.debug("Model performence measures for LinearRegression")
log.debug("LinearRegression_MAE:", lin_mae)
log.debug("LinearRegression_MSE:", lin_mse)
log.debug("LinearRegression_RMSE:", lin_rmse)


# Prediction using DecisionTreeRegressor Model
tree_reg_housing_predictions = tree_reg.predict(test)

tree_mae = mean_absolute_error(test_labels, tree_reg_housing_predictions)
tree_mse = mean_squared_error(test_labels, tree_reg_housing_predictions)
tree_rmse = np.sqrt(tree_mse)

log.debug("Model performence measures for DecisionTreeRegressor")
log.debug("DecisionTreeRegressor_MAE:", tree_mae)
log.debug("DecisionTreeRegressor_MSE:", tree_mse)
log.debug("DecisionTreeRegressor_RMSE:", tree_rmse)


# Prediction using RandomForestRegressor Model
final_forest_reg_housing_predictions = final_forest_reg_model.predict(test)

Forest_mae = mean_absolute_error(test_labels, final_forest_reg_housing_predictions)
forest_mse = mean_squared_error(test_labels, final_forest_reg_housing_predictions)
forest_rmse = np.sqrt(forest_mse)

log.debug("Model performence measures for RandomForestRegressor")
log.debug("RandomForestRegressor_MAE:", Forest_mae)
log.debug("RandomForestRegressor_MSE:", forest_mse)
log.debug("RandomForestRegressor_RMSE:", forest_rmse)

# As the errors are less for RandomForestRegressor we are using this model's predictions in output file.

Prediction_output = test.copy()
Prediction_output["Predicted House Price"] = final_forest_reg_housing_predictions

# Writing the file with predicted house price
Prediction_output.to_csv(os.path.join(output_path, "Prediction_output.csv"), index=False)
log.debug("Prediction file is created")

printout()
