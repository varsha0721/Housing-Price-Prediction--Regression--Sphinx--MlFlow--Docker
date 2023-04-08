from setuptools import setup

setup(
    name="pack_install",
    version="1.0",
    description="This package tells if the given package is installed or not",
    long_description="The package takes the list of package and returns an dictonery with two keys and two lists as values for the respective keys.The two lists contains the list of installed and not installed packages.",
    author="Varsha Rajawat",
    author_email="varsha.rajawat@tigeranalytics.com",
    install_requires=["importlib"],
)
