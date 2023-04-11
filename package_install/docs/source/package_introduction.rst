*****************************************
Getting started with pack_install Package
*****************************************

This project aims at identifing the installed and uninstalled packages

Following is the function that we have developed in Python

    - Check packages are installed or not

Find installed and uninstalled pcakages
========================================

In  this function we take one input parameter, a list of package names which we want to check are installed or not.
The data type can be only ``list``.
Input ``list`` should not be empty.

.. code-block:: python
    :emphasize-lines: 2, 3, 13, 14, 18, 20
    :linenos:
    :caption: Code to find the installed and uninstalled packages.

    def package_installation_status(lst):
    if type(lst) == list:                                            # Check if input parameter is list or not.
        if len(lst) != 0:                                            # Check if list is empty or not
            installed_package_list = []
            unistalled_package_list = []
            for name in lst:
                spec = importlib.util.find_spec(name)
                if spec is None:
                    unistalled_package_list.append(name)
                else:
                    installed_package_list.append(name)
            package_info = {
                "installed_packages": installed_package_list,         # Creating the list of installed packages.
                "not_installed_packages": unistalled_package_list,    # Creating the list of uninstalled packages.
            }
            return package_info
        else:
            raise ValueError("The list is empty")                     # Value Error will be raised if list is empty.
    else:
        raise TypeError("Please provide list input")                  # Type Error will be raised if datatype is not list.

From the code block we see that this code takes necessary input parameter, a list of package names and returns a dictonery
containing two lists of installed ans not installed packages.

Examples
--------
Below are some examples of running the code with different inputs.

- When some of the given packages are installed and some not ::

.. code-block:: python
    :emphasize-lines: 1
    :linenos:
    :caption: NO ERROR

    >>> print(package_installation_status(["numpy","pandas","matplotlib","tensorflow"]))
    {'installed_packages': ['numpy', 'pandas'], 'not_installed_packages': ['matplotlib', 'tensorflow']}

- When all the given packages are installed ::

.. code-block:: python
    :emphasize-lines: 1
    :linenos:
    :caption: NO ERROR

    >>> print(package_installation_status(["numpy","pandas"]))
    {'installed_packages': ['numpy', 'pandas'], 'not_installed_packages': []}

- When all the given packages are not installed ::

.. code-block:: python
    :emphasize-lines: 1
    :linenos:
    :caption: NO ERROR

    >>> print(package_installation_status(["matplotlib","tensorflow"]))
    {'installed_packages': [], 'not_installed_packages': ['matplotlib', 'tensorflow']}

- When input list is empty::

.. code-block:: python
    :emphasize-lines: 1
    :linenos:
    :caption: ERROR

    >>> print(package_installation_status([]))
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        File "<stdin>", line 18, in package_installation_status
    ValueError: The list is empty

- When input Parameter is other than **list**::

.. code-block:: python
    :emphasize-lines: 1
    :linenos:
    :caption: ERROR

    >>> print(package_installation_status("pandas"))
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        File "<stdin>", line 20, in package_installation_status
    TypeError: Please provide list as input

- When no input is given::

.. code-block:: python
    :emphasize-lines: 1
    :linenos:
    :caption: ERROR

    >>> print(package_installation_status())
        Traceback (most recent call last):
        TypeError: package_installation_status()
    missing 1 required positional argument: 'lst'