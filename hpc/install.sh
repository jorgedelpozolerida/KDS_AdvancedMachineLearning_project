#!/bin/bash

# Read the list of packages from requirements.txt
packages=$(cat requirements.txt)

# Install each package and print a message to the terminal
for package in $packages
do
    echo -e "\n\nInstalling package $package\n"
    pip install $package
done