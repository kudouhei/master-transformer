#! /bin/bash

# Fix for active issues
eval "$(command conda 'shell.bash' 'hook' 2>/dev/null)"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found"
    exit 1
fi

# Check for git lfs
if ! command -v git-lfs &> /dev/null; then
  echo "please install git-lfs, e.g. 'brew install git-lfs' or visit https://git-lfs.com/ for your OS instructions"
  exit 1
fi

# check if the 'llmpro' environment exists
if conda info --envs | grep -q 'llmpro'; then
    echo "llmpro environment already exists"
else
    echo "creating llmpro environment"
    conda create -n llmpro python=3.10 --yes
fi

conda activate llmpro

# Install the required packages
echo "Installing requirements from requirements.txt:"
pip install -r requirements.txt

# Deactivate the conda environment after installing packages
echo "Packages installed. Deactivating conda environment..."
conda deactivate

# Create an environment for each directory
for dir in src/*/; do
    dir_name=$(basename "$dir")
    echo "Creating environment for $dir_name"

    # check for the requirements.txt file
    if [ ! -f "$dir/requirements.txt" ]; then
        echo "No requirements.txt file found for $dir_name"
        continue
    fi

    # check if the environment already exists
    if conda info --envs | grep -q "$dir_name"; then
        echo "Environment $dir_name already exists. Skipping environment creation."
    else
        echo "Creating environment for $dir_name"
        conda create -n "$dir_name" python=3.10 --yes
    fi

    # activate the environment
    conda activate "$dir_name"

    # install the requirements
    pip install -r "$dir/requirements.txt"
    
    # deactivate the environment
    echo "Packages installed. Deactivating conda environment..."
    conda deactivate
    
done

echo -e "\n\n\n"
echo -e "You are now set up.\n\nPlease run 'conda activate 1_basic' to begin.\n"