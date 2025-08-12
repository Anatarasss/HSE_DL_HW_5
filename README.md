# Deep Learning Model Interpretability

This project demonstrates various approaches for interpreting predictions of deep learning models, with a focus on computer vision tasks using pretrained models.

## Project Overview

The notebook explores two main interpretability techniques:

1. **SmoothGrad** - A gradient-based explanation method that averages gradients over noisy inputs to highlight important pixels.
2. **SHAP (SHapley Additive exPlanations)** - A game-theoretic approach to explain model outputs by computing feature importance values.

The project uses pretrained models (DenseNet121 and ResNet50) on ImageNet and shows how to generate explanations for their predictions.

## Key Features

- Implementation of SmoothGrad from scratch
- Application of SHAP's GradientExplainer for CNN interpretability
- Comparison of explanations at different network depths (layers 2 and 4)
- Examples with various images including scientific figures and portraits
- Extension to classical ML models (CatBoost) using TreeExplainer

## Tasks Implemented

1. **Manual SmoothGrad Implementation**:
   - Creating noisy input variations
   - Computing and averaging gradients
   - Visualizing explanations for multiple classes

2. **Intermediate Layer Explanations**:
   - Using SHAP to explain ResNet50 predictions
   - Comparing explanations from different network depths
   - Analyzing how feature importance changes across layers

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- scikit-image
- shap
- catboost (for classical ML examples)
- matplotlib

## Usage

The notebook is designed to work in Jupyter environment. Key sections:

1. Load pretrained models and example images
2. Generate SmoothGrad explanations
3. Compute SHAP values using GradientExplainer
4. Visualize explanations for different classes
5. Compare layer-wise explanations

## Examples Included

- Nobel prize winner photos
- Scientific equipment images
- Classical art portraits
- Boston housing dataset (for classical ML)

## References

- SHAP documentation and examples
- Original SmoothGrad and SHAP papers
- ELI5 interpretability package