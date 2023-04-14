import argparse
import os
import tarfile
from urllib.parse import urlparse

import mlflow
from logging_tree import printout
from pack_install import find_installed_package as pk
from six.moves import urllib

from log_config import configure_logger

# Configuring logger
log_file = False
Console = True
Log_level = "DEBUG"
log = configure_logger(log_file, Console, Log_level)

DOWNLOAD_ROOT = (
    "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
)

# Testing the package pack_install
def package_install_status():
    required_package_list = ["argparse", "os", "tarfile", "six.moves"]
    Package_status = pk.package_installation_status(required_package_list)
    log.debug(Package_status)
    mlflow.log_param("ingest_data_Package_status", Package_status)


# Function to write unit test cases
def add(x, y):
    """Add Function"""
    return x + y


sum = add(2, 5)


# Fetching raw data
def fetch_housing_data(housing_url, RAW_PATH, PROCESS_PATH):
    tgz_path = os.path.join(RAW_PATH, "housing.tgz")
    # urllib.request.urlretrieve(housing_url, tgz_path)
    log.debug("raw data downloaded")
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(PROCESS_PATH)
    log.debug("raw data is converted to the .csv file")
    housing_tgz.close()


def ingest_data_main():
    RAW_PATH = "./data/raw"
    PROCESS_PATH = "./data/process"

    # Calling all the methods
    package_install_status()
    fetch_housing_data(DOWNLOAD_ROOT, RAW_PATH, PROCESS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Takes the name of directories to store the datasets/files"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="add the root directory to keep all the data sets, default name of direactory is 'data'",
        default="data",
    )
    parser.add_argument(
        "--raw",
        type=str,
        help="add the directory inside parent directory to keep all the raw datasets, default name of direactory is 'raw'",
        default="raw",
    )
    parser.add_argument(
        "--process",
        type=str,
        help="add the directory inside parent directory to keep all the processed datasets, default name of direactory is  'process'",
        default="process",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="add the directory inside parent directory to keep all the files for output, default name of direactory is  'output'",
        default="output",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        help="add the directory to keep project artifacts 'artifacts'",
        default="artifacts",
    )
    args = parser.parse_args()
    data = args.data
    raw = args.raw
    process = args.process
    output = args.output
    artifacts = args.artifacts

    RAW_PATH = os.path.join(data, raw)
    PROCESS_PATH = os.path.join(data, process)
    OUTPUT_PATH = os.path.join(data, output)
    ARTIFACTS_PATH = artifacts

    # Making Directories
    os.makedirs(RAW_PATH, exist_ok=True)
    os.makedirs(PROCESS_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    # MLFLOW
    mlflow.log_param("RAW_PATH", RAW_PATH)
    mlflow.log_param("PROCESS_PATH", PROCESS_PATH)
    mlflow.log_param("OUTPUT_PATH", OUTPUT_PATH)
    mlflow.log_param("ARTIFACTS_PATH", ARTIFACTS_PATH)

    ingest_data_main()
    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    printout()
