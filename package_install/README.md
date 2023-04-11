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

## Commands to create the package
 -  python -m build
 .whl and .tar.gz files will be created from above command.