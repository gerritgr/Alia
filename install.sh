#!/bin/bash

# Do not run as root

sudo apt-get install nvidia-modprobe -y 

# Step 0: Clone the GitHub repository and change directory
echo "Cloning the Alia repository..."
#git clone https://github.com/gerritgr/Alia.git
# cd Alia

echo "Changed directory to Alia."

# Create api_key.txt with specified content
echo "0a636acbfb73590367d696af485c4d032833f16d" > api_key.txt
echo "api_key.txt created with content."

# Step 1: Download and Install Anaconda
# Replace the URL with the latest Anaconda installer link for Linux
echo "Downloading Anaconda..."
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O anaconda.sh

echo "Installing Anaconda..."
bash anaconda.sh -b -p $HOME/anaconda
echo "Anaconda installed."

exec bash

# Initialize Anaconda (you may need to restart your shell after running this script)
$HOME/anaconda/bin/conda init

# Step 2: Create and activate environment from environment.txt
# Replace 'myenv' with your desired environment name
ENV_NAME="myenv"
echo "Creating environment: $ENV_NAME"
$HOME/anaconda/bin/conda env create -f environment.txt -n $ENV_NAME

# Wait for the environment to be created
echo "Environment $ENV_NAME created."

# Step 3: Run train.py using the new environment
# This step avoids using 'conda activate' by specifying the environment's Python executable directly
echo "Running train.py with $ENV_NAME environment..."
$HOME/anaconda/envs/$ENV_NAME/bin/python train.py

echo "Script completed."
