#! /bin/bash

# check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found"
    exit 1
fi

echo "Removing all conda environments"

# remove the main env
conda env remove -n llmpro

# remove all other envs
for dir in src/*/; do
    dir_name=$(basename "$dir")
    conda env remove -n "$dir_name"
done

echo "All conda environments removed"