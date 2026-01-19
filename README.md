# Convolution Filter Neural Network - Textbook Implementation

This project implements convolutional neural network models based on the textbook **Generative Deep Learning (2nd Edition)** by David Foster, specifically from Chapter 2 on Convolutions.

## Overview

This repository contains two CNN models that demonstrate the effect of convolution stride on neural network performance:

- **Model 1**: Original Convolution Model (Textbook Version) - Baseline CNN with stride = 1
- **Model 2**: Modified Convolution Model - Explicitly uses stride = 1 for comparison

Both models are trained on the MNIST handwritten digit dataset to classify digits 0-9.

## Project Structure

```
convolutionFilter-NN/
├── original_convolution_model.ipynb  # Model 1: Original baseline CNN (Jupyter Notebook)
├── original_convolution_model.py    # Model 1: Original baseline CNN (Python Script)
├── modified_convolution_model.ipynb  # Model 2: Modified CNN with stride = 1 (Jupyter Notebook)
├── modified_convolution_model.py    # Model 2: Modified CNN with stride = 1 (Python Script)
├── requirements.txt                 # Python dependencies
├── SETUP.md                         # Detailed setup instructions
└── README.md                        # This file
```

**Note**: Both Jupyter notebooks (`.ipynb`) and Python scripts (`.py`) are provided. You can use either format based on your preference.

## Model Architecture

Both models use the same architecture:

- **Input**: 28×28 grayscale images (MNIST digits)
- **Conv2D Layer**: 
  - 32 filters
  - 3×3 kernel size
  - Stride = (1, 1)
  - Valid padding
  - ReLU activation
- **Flatten Layer**: Converts 2D feature maps to 1D vector
- **Dense Layer**: 32 units with ReLU activation
- **Output Layer**: 10 units (one per digit class) with softmax activation

**Total Parameters**: 692,906 (2.64 MB)

## Requirements

- Python 3.11 (TensorFlow doesn't support Python 3.14+)
- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- Jupyter Notebook or JupyterLab (optional, for running `.ipynb` files)

## Setup Instructions

### Option 1: Using Conda (Recommended)

1. Create a conda environment with Python 3.11:
   ```bash
   conda create -n convfilter python=3.11 -y
   ```

2. Activate the environment:
   ```bash
   conda activate convfilter
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using System Python

If you have Python 3.11 installed system-wide:

```bash
pip install -r requirements.txt
```

**Note**: TensorFlow requires Python 3.7-3.12. Python 3.14 is not supported.

## Running the Models

You can run the models using either **Jupyter Notebooks** (recommended for interactive exploration) or **Python scripts** (for command-line execution).

### Option 1: Using Jupyter Notebooks (Recommended)

1. **Start Jupyter Notebook**:
   ```bash
   conda activate convfilter
   jupyter notebook
   ```
   Or use JupyterLab:
   ```bash
   conda activate convfilter
   jupyter lab
   ```

2. **Open the notebook**:
   - `original_convolution_model.ipynb` for Model 1
   - `modified_convolution_model.ipynb` for Model 2

3. **Select the correct kernel** (IMPORTANT):
   - When you open the notebook, you may see a kernel selection prompt
   - **Select "Python 3.11 (convfilter)"** from the kernel dropdown
   - If you don't see this option, go to: Kernel → Change Kernel → Python 3.11 (convfilter)
   - **Do NOT use Python 3.13** - TensorFlow doesn't support it!

4. **Run all cells**: Use "Run All" from the Cell menu, or run cells individually with Shift+Enter

**Note**: The kernel "Python 3.11 (convfilter)" has been pre-installed. If you don't see it, you may need to restart Jupyter after activating the conda environment.

### Option 2: Using Python Scripts

**Model 1: Original Convolution Model**

```bash
# If using conda environment
conda activate convfilter
python original_convolution_model.py

# Or use full path
/opt/anaconda3/envs/convfilter/bin/python original_convolution_model.py
```

**Model 2: Modified Convolution Model**

```bash
# If using conda environment
conda activate convfilter
python modified_convolution_model.py

# Or use full path
/opt/anaconda3/envs/convfilter/bin/python modified_convolution_model.py
```

### Installing Jupyter (if needed)

If you want to use the notebooks, install Jupyter:

```bash
conda activate convfilter
pip install jupyter notebook ipykernel
# Or for JupyterLab:
pip install jupyterlab ipykernel
```

**Important**: The kernel "Python 3.11 (convfilter)" has been registered. When opening notebooks, make sure to select this kernel (not Python 3.13) to ensure TensorFlow compatibility.

## Expected Output

Each notebook/script will output:

1. **Dataset Information**: Training, validation, and test set sizes
2. **Model Summary**: Layer-by-layer architecture with parameter counts
3. **Convolution Layer Configuration**: 
   - Filters, kernel size, stride, padding
   - Input/output shapes
4. **Training Progress**: Per-epoch training and validation accuracy
5. **Final Results**: 
   - Final training accuracy
   - Final validation accuracy
   - Final test accuracy
   - Final test loss

### Example Output Format

```
======================================================================
MODEL 1: ORIGINAL CONVOLUTION (TEXTBOOK VERSION)
======================================================================
Final Training Accuracy:   0.9918
Final Validation Accuracy: 0.9803
Final Test Accuracy:       0.9805
Final Test Loss:           0.0674
======================================================================
```

## Model Comparison

Both models are architecturally identical (both use stride = 1). Model 2 is provided as a separate implementation to explicitly demonstrate the stride parameter setting.

### Key Differences in Documentation:

- **Model 1**: Represents the baseline/original convolution model from the textbook
- **Model 2**: Explicitly sets and documents stride = 1 to show how stride affects feature maps

### Expected Performance:

Both models should achieve similar results:
- **Test Accuracy**: ~97-98%
- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~97-98%

## Understanding the Models

### Convolution Stride

- **Stride = 1**: Filter moves 1 pixel at a time, preserving maximum spatial information
  - Input: 28×28 → Output: 26×26 (with 3×3 kernel, valid padding)
  - Preserves fine-grained spatial features
  
- **Stride = 2**: Filter moves 2 pixels at a time, reducing feature map size
  - Input: 28×28 → Output: 13×13 (with 3×3 kernel, valid padding)
  - Reduces computational cost but loses spatial detail

### Why Stride Matters

- **Stride = 1**: 
  - Examines every pixel position
  - Preserves maximum spatial resolution
  - Better for fine-grained feature detection
  - Larger feature maps (more parameters in dense layers)

- **Larger Stride (e.g., 2)**:
  - Skips pixel positions
  - Reduces feature map size
  - Faster computation, fewer parameters
  - May lose important spatial details

## Academic Use

This implementation is designed for academic submission. 

- **Jupyter Notebooks**: Perfect for interactive exploration, visualization, and step-by-step execution. You can export notebooks to PDF or HTML for submission.
- **Python Scripts**: Ideal for command-line execution and automated runs. All outputs are formatted as plain text that can be directly copied into Word documents or reports.

### What to Include in Your Report:

1. Model architecture summary
2. Convolution layer configuration (filters, kernel size, stride, padding)
3. Training/validation accuracy per epoch
4. Final test accuracy and loss
5. Comparison between models (if applicable)
6. Discussion of how stride affects feature map size and model performance

## Troubleshooting

### Issue: TensorFlow Installation Fails

**Problem**: `ERROR: Could not find a version that satisfies the requirement tensorflow`

**Solution**: 
- Ensure you're using Python 3.11 or earlier (not 3.14+)
- Use the conda environment as described in SETUP.md
- Verify Python version: `python --version`

### Issue: ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
- Activate the conda environment: `conda activate convfilter`
- Or install dependencies: `pip install -r requirements.txt`

### Issue: Keras Warning About input_shape

**Problem**: Warning about `input_shape` argument in Sequential models

**Solution**: This is a harmless warning in Keras 3.x. The models will still train correctly. You can ignore this warning.

### Issue: Wrong Python Version in Jupyter Notebook

**Problem**: Notebook is using Python 3.13 (or another version) instead of Python 3.11

**Solution**: 
1. Make sure you've activated the conda environment: `conda activate convfilter`
2. In the notebook, go to: **Kernel → Change Kernel → Python 3.11 (convfilter)**
3. If the kernel doesn't appear, restart Jupyter after activating the environment
4. Verify the kernel by running: `import sys; print(sys.version)` - should show Python 3.11.x

## References

- **Textbook**: Generative Deep Learning (2nd Edition) by David Foster
- **Notebook**: https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/02_deeplearning/02_cnn/convolutions.ipynb
- **Dataset**: MNIST Handwritten Digits (built into TensorFlow/Keras)

## License

This implementation is for educational purposes, based on the textbook examples.

## Contact

For questions or issues, please refer to the textbook or the original GitHub repository.

