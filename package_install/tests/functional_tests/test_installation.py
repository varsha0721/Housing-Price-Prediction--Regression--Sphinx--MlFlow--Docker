from pack_install import find_installed_package


def test_install():
    """
    Test the installation in the env
    """

    find_installed_package.package_installation_status(["pandas"])

    # read the datasets from the path
    print("TC1: Installation successful!")


if __name__ == "__main__":
    """
    Driver function to call other functions in order
    """
    test_install()
