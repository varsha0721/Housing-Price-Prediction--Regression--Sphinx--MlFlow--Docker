# Description
 - This package tells if the given package is installed in your environment or not.

## Steps performed
 - Check if the input given is list or not. If not, it returns the TypeError
 - Check if the input list is empty or not. If list is empty then it returns the ValueError
 - After above checks it tests which packages are installed in environment or not.
 - Creates the two lists of installed packages and not installed packages and place them in the dictonery.
 - returns dictonery of keys containing lists of installed packages and not installed packages.

### Input
 - a list

### Output
 - a dictonery

## Info
 - Package Name = pack_install
 - Module Name = find_installed_package

# Make sure you have upgraded version of pip
Windows
```
py -m pip install --upgrade pip
```

Linux/MAC OS
```
python3 -m pip install --upgrade pip
```

## Create a project with the following structure
```
package_install/
├── LICENSE
├── pyproject.toml
├── README.md
├── setup.py
├── src/
│   └── pack_install/
│       ├── __init__.py
│       └── find_installed_package.py
└── tests/
touch LICENSE
touch pyproject.toml
touch setup.py
mkdir src/pack_install
touch src/pack_install/__init__.py
touch src/pack_install/find_installed_package.py
mkdir tests
```

## pyproject.toml

This file tells tools like pip and build how to create your project

```
[build-system]
requires = [
    "setuptools>=42",
    "wheel"
]
build-backend = "setuptools.build_meta"
```
build-system.requires gives a list of packages that are needed to build your package. Listing something here will only make it available during the build, not after it is installed.

build-system.build-backend is the name of Python object that will be used to perform the build. If you were to use a different build system, such as flit or poetry, those would go here, and the configuration details would be completely different than the setuptools configuration described below.


# Setup.py setup
A dynamic setup file using setup.py

```
setup(
    name="pack_install",
    version="1.0",
    description="This package tells if the given package is installed or not",
    long_description="The package takes the list of package and returns an dictonery with two keys and two lists as values for the respective keys.The two lists contains the list of installed and not installed packages.",
    author="Varsha Rajawat",
    author_email="varsha.rajawat@tigeranalytics.com",
    install_requires=["importlib"],
)

```
# Running the build
### Make sure your build tool is up to date
Windows
```
py -m pip install --upgrade build
```
Linux/MAC OS
```
python3 -m pip install --upgrade build
```


### Create the build
```
py -m build
```
.whl and .tar.gz files will be created from above command.

### Uploading the distribution archives
Install Twine:
- py -m pip install --upgrade twine

Once installed, run Twine to upload all of the archives under dist:
- py -m twine upload --repository testpypi dist/* --verbose


### Installing your newly uploaded package
- pip install -i https://test.pypi.org/simple/ pack-install==1.0

### My Token
pypi-AgENdGVzdC5weXBpLm9yZwIkMjI3YzU0MzQtYjhkMi00MDg3LTk4NDAtMTczMThmYjcxZTIzAAIqWzMsIjhkZmQyZDdjLTQ4ZjMtNGNlNS1iOWI4LWZkMjlhNTI0NWU2MCJdAAAGIBhdxek_OyzKtEDav-7wLqEpLS5M_lnNVWTqjgATRHzF

### References
https://packaging.python.org/tutorials/packaging-projects/
