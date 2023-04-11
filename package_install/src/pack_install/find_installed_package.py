import importlib.util


def package_installation_status(lst):
    """This package tells if the given package is installed or not according to list of name of packages given in the argument ``lst``.

    Args:
        lst (list, positional):  List containing the name of packages to check if the given package is installed or not.::

    Returns:
        Dictonery

        Key1 : installed_packages

        Value1 : list of all the packages installed in your environment.


        Key2 : not_installed_packages

        Value2 : list of all the packages installed in your environment.

    Raises:
        TypeError
            If ``lst`` datatype is not list
        ValueError
            If ``lst`` is empty.

    .. note::

        There are no default values if arguments are not given, it will raise TypeError.
    """

    if type(lst) == list:
        if len(lst) != 0:
            installed_package_list = []
            unistalled_package_list = []
            for name in lst:
                spec = importlib.util.find_spec(name)
                if spec is None:
                    unistalled_package_list.append(name)
                else:
                    installed_package_list.append(name)
            package_info = {
                "installed_packages": installed_package_list,
                "not_installed_packages": unistalled_package_list,
            }
            return package_info
        else:
            raise ValueError("The list is empty")
    else:
        raise TypeError("Please provide list input")
