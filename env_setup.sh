#!/bin/bash
# Activate the environment using the symbo-linked directory
source ~/ECON_env/myenv/bin/activate
echo "Environment setup complete"

# Start an interactive shell to keep the GPU allocation running
exec bash --login
