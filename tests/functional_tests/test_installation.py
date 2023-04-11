import sys

import pandas as pd

sys.path.insert(0, "../TCE_2023")

data_path = "data/process"

housing = pd.read_csv(data_path + "/housing.csv")
test = pd.read_csv(data_path + "/test.csv")

housing.head()
def test_install():
    """
    Test the installation in the env
    """
    # read the datasets from the path
    print("TC1: Installation successful!")


def test_rows():
    """
    Test the correctness of the rows of data
    """
    assert housing.shape[0] == 20640
    assert test.shape[0] == 4128
    print(housing.shape)

    print("TC2: Rows verified successfully!")


def test_columns():
    """
    Test the correctness of the columns of data
    """
    assert housing.shape[1] == 10
    assert test.shape[1] == 15

    print("TC3: Columns verified successfully!")


if __name__ == "__main__":
    """
    Driver function to call other functions in order
    """
    test_install()
    test_rows()
    test_columns()
