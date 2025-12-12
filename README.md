# finding_exoplanet

**Author**: Chikara Oe  
**Last Updated**: December 11, 2025

## ML Part

### Overview
The ML pipeline supports two preprocessing versions (V2 and V4). Both use the same CNN architecture, but differ in data preprocessing methods.

### Data Download

Before running preprocessing, download the Kepler exoplanet dataset:

```bash
python downloader.py
```

This script downloads the dataset from Kaggle and copies the KOI (Kepler Object of Interest) CSV file to the project directory.

### Preprocessing Options

#### V4 (Recommended - Latest)
1. Run preprocessing: `Zack_data_cleaning_pipeline_v4.ipynb`
2. Run ML pipeline: `ml_pipeline_for_V4_preprocessing.py`

#### V2 (Legacy)
1. Run preprocessing: `preprocessing_V2.py`
2. Run ML pipeline: `ml_pipeline_for_V2_preprocessing.ipynb`

**Note on V2**: Training curves show good performance, but validation curves indicate overfitting (loss increases, accuracy decreases).

<img src="result_of_V2.png" alt="V2 Training Results" width="600"/>

*V2 preprocessing results showing overfitting: training loss/accuracy improve, but validation loss increases and accuracy decreases.*

### Model Architecture

The CNN model uses a dual-column architecture that processes both local (zoomed-in) and global (full phase-folded) views of light curves in parallel, then combines their features for binary classification.

#### Architecture Overview

```
Input: (batch_size, 1, 201) [Local] + (batch_size, 1, 2001) [Global]
    ↓
[Local CNN Column]          [Global CNN Column]
    ↓                            ↓
Flatten (1,504)          Flatten (15,744)
    ↓                            ↓
    └────────── Concatenate (17,248) ──────────┘
                        ↓
                [MLP Head: 5 FC Layers]
                        ↓
              Output: (batch_size, 1)
```

#### Local CNN Column

Processes the zoomed-in transit view to capture fine-grained transit features.

- **Input**: `(batch_size, 1, 201)` - Local flux data around transit
- **local_conv1**:
  - Conv1d: `1 → 16 channels, kernel_size=5, padding=0`
  - ReLU activation
  - Conv1d: `16 → 16 channels, kernel_size=5, padding=0`
  - ReLU activation
  - MaxPool1d: `kernel_size=7, stride=2`
- **Output Shape**: `(batch_size, 16, 94)`
- **Flattened**: `(batch_size, 1,504)` features

#### Global CNN Column

Processes the full phase-folded light curve to capture long-term patterns and context.

- **Input**: `(batch_size, 1, 2001)` - Full phase-folded flux data
- **global_conv1**:
  - Conv1d: `1 → 16 channels, kernel_size=5, padding=0`
  - ReLU activation
  - Conv1d: `16 → 16 channels, kernel_size=5, padding=0`
  - ReLU activation
  - MaxPool1d: `kernel_size=5, stride=2`
- **Output Shape**: `(batch_size, 16, 995)`
- **global_conv2**:
  - Conv1d: `16 → 32 channels, kernel_size=5, padding=0`
  - ReLU activation
  - Conv1d: `32 → 32 channels, kernel_size=5, padding=0`
  - ReLU activation
  - MaxPool1d: `kernel_size=5, stride=2`
- **Output Shape**: `(batch_size, 32, 492)`
- **Flattened**: `(batch_size, 15,744)` features

#### MLP Head

Combines features from both columns and performs binary classification.

- **Concatenation**: Local (1,504) + Global (15,744) = **17,248 features**
- **FC1**: Linear `17,248 → 512` + ReLU
- **FC2**: Linear `512 → 512` + ReLU
- **FC3**: Linear `512 → 512` + ReLU
- **FC4**: Linear `512 → 512` + ReLU
- **FC5**: Linear `512 → 1` (no activation - raw logits)
- **Output**: `(batch_size, 1)` - Binary classification logits
- **Loss Function**: BCEWithLogitsLoss (sigmoid applied during loss calculation)

#### Design Rationale

- **Dual-column design**: Separates local transit features from global light curve patterns
- **1D convolutions**: Appropriate for time-series light curve data
- **Progressive downsampling**: MaxPool layers reduce dimensionality while preserving important features
- **Feature fusion**: Concatenation allows the model to learn interactions between local and global patterns

### Usage

```python
# V4 (Recommended)
python ml_pipeline_for_V4_preprocessing.py

# V2 (Legacy)
# Run ml_pipeline_for_V2_preprocessing.ipynb
```