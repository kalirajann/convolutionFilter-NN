# Setup Instructions

## Issue: Python 3.14 Compatibility

TensorFlow doesn't support Python 3.14 yet. TensorFlow supports Python 3.7-3.12.

## Solution: Use the Conda Environment (Already Created)

A conda environment named `convfilter` with Python 3.11 has been created and TensorFlow is already installed.

## To Use the Environment

**Option 1: Activate the environment in your terminal**
```bash
conda activate convfilter
python original_convolution_model.py
```

**Option 2: Use the full path to the Python interpreter**
```bash
/opt/anaconda3/envs/convfilter/bin/python original_convolution_model.py
```

## Dependencies

All dependencies are already installed:
- TensorFlow 2.20.0
- NumPy 2.4.1
- Keras 3.13.1
- And all required dependencies

## Running the Script

Once you're in the conda environment, simply run:
```bash
python original_convolution_model.py
```

