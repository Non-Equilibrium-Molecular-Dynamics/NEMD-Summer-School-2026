#!/usr/bin/env bash

# Python setup script for the NEMD Summer School on Cirrus.
# Run this script with:
#
#   source ./python_setup.sh


module load PrgEnv-gnu
module load cray-python

VENV_DIR="/work/tc075/tc075/${USER}/myvenv"

echo "Creating Python virtual environment in:"
echo "$VENV_DIR"

python -m venv --system-site-packages "$VENV_DIR"

echo "Activating Python virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing Python packages..."
python -m pip install matplotlib scikit-learn

echo "Creating test.py..."
cat > test.py <<'EOF'
import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.sin(np.linspace(0,4*np.pi,100)), 'o-')
plt.show()
EOF

echo
echo "Python setup complete."
echo "The virtual environment is now active."
echo "Test it with:"
echo "python test.py"

