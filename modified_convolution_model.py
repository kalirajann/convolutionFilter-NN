#!/usr/bin/env python3
"""
Modified Convolution Model - Stride = 1 Version
Based on Generative Deep Learning (2nd Edition), Chapter 2 - Convolutions

This script implements Model 2: a modified version with stride = 1.
This model is identical to Model 1 in architecture but explicitly uses stride = 1
for the convolution layer to demonstrate the effect of stride on feature maps.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import numpy as np

# ============================================================================
# 1. LOAD AND PREPARE THE DATASET
# ============================================================================
# Using MNIST dataset (handwritten digits) - same as Model 1
# Images are 28x28 grayscale, 10 classes (digits 0-9)
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1] range for better training stability
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Expand dimensions to add channel axis: (60000, 28, 28) -> (60000, 28, 28, 1)
# Conv2D layers require input shape: (height, width, channels)
x_train_full = np.expand_dims(x_train_full, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(f"Training data shape: {x_train_full.shape}")
print(f"Test data shape: {x_test.shape}")

# ============================================================================
# 2. TRAIN / VALIDATION SPLIT
# ============================================================================
# Same split as Model 1: 50,000 samples for training, 10,000 for validation
x_train, x_val = x_train_full[:50000], x_train_full[50000:]
y_train, y_val = y_train_full[:50000], y_train_full[50000:]

print(f"\nTrain set: {x_train.shape[0]} samples")
print(f"Validation set: {x_val.shape[0]} samples")
print(f"Test set: {x_test.shape[0]} samples")

# ============================================================================
# 3. BUILD THE MODIFIED CONVOLUTION MODEL (STRIDE = 1)
# ============================================================================
print("\n" + "=" * 70)
print("BUILDING MODEL 2: MODIFIED CONVOLUTION (STRIDE = 1)")
print("=" * 70)

model = models.Sequential()

# ----------------------------------------------------------------------------
# CONVOLUTION LAYER WITH STRIDE = 1 (MODIFIED FROM ORIGINAL)
# ----------------------------------------------------------------------------
# The Conv2D layer applies learnable filters (kernels) to detect features
# in the input image. Each filter performs a sliding window operation.
#
# KEY MODIFICATION: stride = (1, 1)
# - Stride = 1 means the filter moves 1 pixel at a time in both directions
# - This preserves maximum spatial information from the input
# - With stride = 1 and valid padding, the output feature map size is:
#   output_size = (input_size - kernel_size + 1) / stride
#   For 28x28 input with 3x3 kernel: (28 - 3 + 1) / 1 = 26x26
#
# Why stride = 1 preserves more spatial information:
# - The filter examines every possible position in the input image
# - No information is skipped between positions
# - This results in a larger feature map (26x26) compared to stride = 2 (13x13)
# - More spatial detail is retained, which can help with fine-grained features
#
# Comparison to Model 1:
# - Model 1 also used stride = 1, so this model is architecturally identical
# - However, explicitly setting stride = 1 demonstrates the baseline case
# - If stride were 2, the feature map would be 13x13, losing spatial resolution
# - Stride = 1 is optimal for preserving spatial relationships in images
#
# Key hyperparameters (same as Model 1, except stride is explicitly set to 1):
# - filters=32: Number of different feature detectors (32 different patterns)
# - kernel_size=(3,3): Each filter is 3x3 pixels
# - strides=(1,1): Filter moves 1 pixel at a time (MODIFIED: explicitly stride = 1)
#   * This is the key difference: stride = 1 preserves full spatial resolution
#   * Larger strides (e.g., 2) would reduce feature map size and lose detail
# - padding='valid': No zero-padding around edges
#   * With valid padding and 3x3 kernel, output is 26x26 (28 - 3 + 1 = 26)
# - activation='relu': ReLU activation introduces non-linearity
model.add(layers.Conv2D(filters=32,
                        kernel_size=(3, 3),
                        strides=(1, 1),  # EXPLICITLY SET TO 1
                        padding='valid',
                        activation='relu',
                        input_shape=(28, 28, 1)))

# Flatten the 2D feature maps into a 1D vector for dense layers
# Output from Conv2D: (batch, 26, 26, 32) -> Flatten -> (batch, 26*26*32)
# With stride = 1, we get 26x26 = 676 spatial positions per filter
# Total flattened size: 26 * 26 * 32 = 21,632 features
model.add(layers.Flatten())

# Dense (fully connected) layer for feature combination
# 32 units: same as Model 1 for direct comparison
model.add(layers.Dense(units=32, activation='relu'))

# Output layer: 10 units for 10 digit classes, softmax for probability distribution
model.add(layers.Dense(units=10, activation='softmax'))

# ============================================================================
# 4. COMPILE THE MODEL
# ============================================================================
# Same optimizer and loss as Model 1 for direct comparison
# Using RMSprop optimizer (common baseline choice)
# Sparse categorical crossentropy: appropriate for integer labels (0-9)
# Accuracy metric: tracks classification accuracy during training
model.compile(optimizer=optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ============================================================================
# 5. PRINT MODEL SUMMARY AND CONVOLUTION LAYER CONFIGURATION
# ============================================================================
print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)
model.summary()

# Extract and display convolution layer configuration
print("\n" + "=" * 70)
print("CONVOLUTION LAYER CONFIGURATION")
print("=" * 70)
for layer in model.layers:
    if isinstance(layer, layers.Conv2D):
        print(f"Layer Name: {layer.name}")
        print(f"  Filters: {layer.filters}")
        print(f"  Kernel Size: {layer.kernel_size}")
        print(f"  Strides: {layer.strides}  <-- EXPLICITLY SET TO (1, 1)")
        print(f"  Padding: {layer.padding}")
        # Get activation name safely
        if hasattr(layer.activation, '__name__'):
            activation_name = layer.activation.__name__
        elif callable(layer.activation):
            activation_name = str(layer.activation)
        else:
            activation_name = str(layer.activation)
        print(f"  Activation: {activation_name}")
        # Input/Output shapes (from model summary above):
        # Input: (28, 28, 1) - MNIST images are 28x28 with 1 channel
        # Output: (26, 26, 32) - With 3x3 kernel, stride=1, and valid padding: 28-3+1=26
        # Stride = 1 preserves the maximum spatial resolution possible
        print(f"  Input Shape: (28, 28, 1)")
        print(f"  Output Shape: (26, 26, 32)  [28-3+1=26 with stride=1, valid padding, 32 filters]")
        print(f"\n  NOTE: Stride = 1 means the filter examines every pixel position,")
        print(f"        preserving maximum spatial information in the feature map.")

# ============================================================================
# 6. TRAIN THE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING MODEL 2: MODIFIED CONVOLUTION (STRIDE = 1)")
print("=" * 70)
print("Training with the following settings:")
print("  - Epochs: 5")
print("  - Batch Size: 64")
print("  - Optimizer: RMSprop")
print("  - Loss: Sparse Categorical Crossentropy")
print("  - Convolution Stride: (1, 1)  <-- KEY MODIFICATION")
print("=" * 70)

# Train the model
# verbose=1: Shows progress bar and metrics for each epoch
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Extract final training and validation accuracies
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

# Print per-epoch accuracies
print("\n" + "=" * 70)
print("TRAINING HISTORY - ACCURACY PER EPOCH")
print("=" * 70)
for epoch in range(len(history.history['accuracy'])):
    print(f"Epoch {epoch + 1}:")
    print(f"  Training Accuracy:   {history.history['accuracy'][epoch]:.4f}")
    print(f"  Validation Accuracy: {history.history['val_accuracy'][epoch]:.4f}")

# ============================================================================
# 7. EVALUATE ON TEST SET
# ============================================================================
print("\n" + "=" * 70)
print("EVALUATING ON TEST SET")
print("=" * 70)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

# ============================================================================
# 8. PRINT FINAL RESULTS (FORMATTED FOR ACADEMIC SUBMISSION)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 2: MODIFIED CONVOLUTION (STRIDE = 1)")
print("=" * 70)
print(f"Final Training Accuracy:   {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Final Test Accuracy:       {test_acc:.4f}")
print(f"Final Test Loss:           {test_loss:.4f}")
print("=" * 70)

print("\nModel Configuration Summary:")
print("  - Architecture: Conv2D -> Flatten -> Dense -> Dense")
print("  - Conv2D: 32 filters, 3x3 kernel, stride = (1, 1), valid padding")
print("  - Stride = 1 preserves maximum spatial resolution (26x26 feature maps)")
print("  - This model is architecturally identical to Model 1")
print("  - Stride = 1 allows the filter to examine every pixel position")
print("  - No pooling, no batch normalization, minimal architecture")
print("=" * 70)

print("\nComparison Notes:")
print("  - Stride = 1 means the convolution filter moves 1 pixel at a time")
print("  - This preserves more spatial information than larger strides")
print("  - Feature map size: 26x26 (with 3x3 kernel and valid padding)")
print("  - If stride were 2, feature map would be 13x13, losing spatial detail")
print("  - Stride = 1 is optimal for preserving fine-grained image features")
print("=" * 70)

