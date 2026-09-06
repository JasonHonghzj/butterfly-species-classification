# Butterfly Species Classification with PyTorch

An end-to-end computer vision project for multi-class butterfly species classification using iNaturalist image data and a custom convolutional neural network (CNN) built with PyTorch.

## Overview

This project develops a complete deep-learning pipeline for classifying butterfly observations into **10 classes: the nine most frequent species in the training data plus an `Other` class**.

The workflow covers exploratory data analysis, label engineering, image preprocessing, custom PyTorch data loading, CNN architecture design, GPU-accelerated training, validation-based model selection, and probabilistic inference on unseen test images.

The final model achieved **79.22% validation accuracy** with a **validation loss of 0.6141** at the best checkpoint.

## Key Results

| Metric | Result |
| --- | ---: |
| Number of classes | **10** |
| Best epoch | **25** |
| Training accuracy | **78.44%** |
| Validation accuracy | **79.22%** |
| Training loss | **0.6391** |
| Validation loss | **0.6141** |
| Input resolution | **224 × 224 RGB** |
| Batch size | **64** |

The test split does not provide ground-truth labels in the project workflow. Therefore, test accuracy is not reported; instead, the final model generates class-probability predictions for each unseen test image.

## Problem Formulation

The original butterfly data contains many species with a long-tailed class distribution. Modeling every rare species independently would create a highly fragmented classification problem.

To create a more practical target:

1. identify the nine most frequent species in the training data;
2. preserve each of those species as an individual class;
3. combine all remaining species into an `Other` category.

This produces a 10-class image-classification problem while retaining the most strongly represented species.

## Project Pipeline

```text
Raw Metadata + Butterfly Images
              |
              v
     Class Distribution EDA
              |
              v
     Top-9 + Other Labeling
              |
              v
      Processed Label Data
              |
              v
     Custom PyTorch Dataset
              |
              v
Preprocessing / Data Augmentation
              |
              v
         DataLoader
              |
              v
        Custom CNN
              |
              v
 AMP + Adam + Gradient Clipping
              |
              v
 Validation + Early Stopping
              |
              v
      Best Model Checkpoint
              |
              v
 Test-Set Probability Predictions
```

## Data

The project uses butterfly observations derived from **iNaturalist**, with metadata describing observation identifiers, image paths, species labels, and dataset splits.

Only the lightweight tabular metadata and processed outputs are included in this repository. The full image collection is substantially larger and is not stored in the repository.

### Repository data organization

```text
data/
├── raw/
│   ├── train.csv
│   ├── test.csv
│   └── train.tsv
│
└── processed/
    ├── train-onehot.csv
    ├── validation-onehot.csv
    └── test-predictions.csv
```

`data/raw/` contains the original tabular metadata used by the project, while `data/processed/` contains transformed labels and the final test-set predictions.

## Image Preprocessing

Images are resized to **224 × 224** pixels and normalized using ImageNet channel statistics:

```python
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

Training-time transformations include image augmentation to improve robustness, while validation and test preprocessing remain deterministic.

A 224 × 224 input resolution was selected after comparing it with 256 × 256 inputs. The larger resolution offered only a limited practical benefit while increasing memory and computation requirements.

## Data Pipeline

A custom `torch.utils.data.Dataset` handles:

- metadata lookup,
- image-path resolution,
- image loading,
- target-label extraction,
- image transformations.

PyTorch `DataLoader`s provide shuffled mini-batches during training and deterministic batching during validation and inference.

This separates data preparation from model training and allows the same preprocessing pipeline to be reused across training, validation, and test data.

## CNN Architecture

The classifier is a custom CNN implemented directly in PyTorch rather than a pretrained image-classification backbone.

```text
Input: 3 × 224 × 224
        |
Conv2D 3 -> 32
BatchNorm + ReLU
        |
Conv2D 32 -> 64
BatchNorm + ReLU
        |
Conv2D 64 -> 128
BatchNorm + MaxPool + ReLU
        |
Conv2D 128 -> 256
BatchNorm + MaxPool + ReLU
        |
Conv2D 256 -> 64
BatchNorm + ReLU
        |
Flatten
        |
Linear -> 256 + ReLU
        |
Linear -> 10 logits
```

### Design choices

- **3 × 3 convolutions** capture local visual patterns while preserving spatial structure with padding.
- Channel depth increases from **32 → 64 → 128 → 256** to learn progressively richer visual representations.
- **Max pooling** reduces spatial dimensionality and computational cost.
- **Batch normalization** improves optimization stability.
- A **256 → 64 bottleneck convolution** compresses the final feature representation.
- The output layer produces 10 logits for multi-class classification.

## Training Strategy

The model is optimized using:

```text
Optimizer: Adam
Learning rate: 5e-4
Weight decay: 5e-4
Loss: Cross-Entropy Loss
Batch size: 64
Maximum epochs: 30
Early-stopping patience: 3
Gradient clipping threshold: 5.0
```

### Mixed-Precision Training

CUDA automatic mixed precision (AMP) is used together with gradient scaling.

This reduces GPU memory usage and can improve training throughput while maintaining numerical stability.

### Gradient Clipping

The global gradient norm is clipped to a maximum value of **5.0** before optimizer updates.

Gradient norms are also monitored during training to help diagnose optimization instability.

### Early Stopping

Validation loss is monitored after every epoch. Training stops when validation performance fails to improve for three consecutive epochs.

The weights corresponding to the best validation loss are retained for final inference.

## Model Performance

The best recorded checkpoint occurred at **epoch 25**:

| Metric | Training | Validation |
| --- | ---: | ---: |
| Loss | **0.6391** | **0.6141** |
| Accuracy | **78.44%** | **79.22%** |

The close training and validation performance suggests that the final training configuration maintained reasonable generalization on the held-out validation set.

## Inference

For unseen test images, the model produces logits for all 10 target classes.

The logits are converted to class probabilities using softmax:

```python
probabilities = torch.softmax(logits, dim=1)
```

The resulting predictions are stored in:

```text
data/processed/test-predictions.csv
```

Each row corresponds to a test observation and contains its identifier together with the predicted probability for each target class.

## Reproducibility

Random seeds are configured for:

- Python `random`,
- NumPy,
- PyTorch,
- CUDA.

Deterministic CuDNN behavior is also enabled where supported to improve reproducibility across repeated experiments.

## Repository Structure

```text
butterfly-species-classification/
├── README.md
├── requirements.txt
├── butterfly_classification.ipynb
│
└── data/
    ├── raw/
    │   ├── train.csv
    │   ├── test.csv
    │   └── train.tsv
    │
    └── processed/
        ├── train-onehot.csv
        ├── validation-onehot.csv
        └── test-predictions.csv
```

### File roles

- `butterfly_classification.ipynb` — exploratory analysis, preprocessing, CNN definition, training, validation, and test inference.
- `data/raw/` — original tabular metadata used by the project.
- `data/processed/` — processed class labels and final model predictions.
- `requirements.txt` — Python dependencies required to run the notebook.
- `README.md` — project documentation.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/butterfly-species-classification.git
cd butterfly-species-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Obtain the image data

The repository contains the tabular metadata and processed outputs, but **does not include the full iNaturalist image dataset** because of its size.

Update the image paths in the notebook as needed for your local or compute environment before running training or inference.

### 4. Run the notebook

```bash
jupyter notebook butterfly_classification.ipynb
```

The notebook contains the complete workflow from exploratory analysis and preprocessing through model training and prediction generation.

## Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **Convolutional Neural Networks**
- **Computer Vision**
- **Pandas**
- **NumPy**
- **Pillow**
- **Matplotlib**
- **CUDA / GPU training**
- **Automatic Mixed Precision (AMP)**
- **Adam optimization**
- **Batch normalization**
- **Gradient clipping**
- **Early stopping**
- **Model checkpointing**
- **Data augmentation**

## What This Project Demonstrates

This project demonstrates an end-to-end applied deep-learning workflow rather than only model fitting.

Key skills include:

- exploratory analysis of class distributions;
- target-label engineering for long-tailed data;
- custom PyTorch `Dataset` and `DataLoader` construction;
- image preprocessing and augmentation;
- custom CNN architecture design;
- GPU-accelerated mixed-precision training;
- optimization monitoring and gradient clipping;
- validation-based model selection;
- reproducible experimentation;
- probabilistic inference on unseen images.

## Limitations

Several limitations should be considered when interpreting the results:

- the model is trained from scratch rather than using a pretrained visual backbone;
- rare species are aggregated into a single `Other` class;
- the project does not report test accuracy because test ground-truth labels are unavailable in the provided workflow;
- the full image dataset is not distributed with this repository;
- validation accuracy alone does not capture class-specific performance, particularly under class imbalance.

## Potential Improvements

Future work could include:

- comparing the custom CNN with transfer-learning baselines such as ResNet or EfficientNet;
- reporting per-class precision, recall, F1-score, and confusion matrices;
- addressing class imbalance with weighted loss or sampling strategies;
- adding learning-rate scheduling;
- performing systematic hyperparameter optimization;
- evaluating calibration of predicted class probabilities;
- refactoring training and inference logic into reusable Python modules;
- adding experiment tracking with MLflow or Weights & Biases.

## Acknowledgements

This project was developed as part of **DS 542 coursework** using butterfly observation data derived from **iNaturalist**.

The repository is presented as a machine-learning portfolio project demonstrating end-to-end computer vision modeling, GPU-accelerated deep-learning training, and reproducible inference with PyTorch.
