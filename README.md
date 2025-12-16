# Exoplanet Detection: Deep Learning & Bayesian Analysis

**Authors:** Iori Adachi, Isabella Deutsch, Levi Galvan, Chikara Oe, Zack Schuder, Linus Upson  
**Course:** Physics 188/288: Bayesian Data Analysis & Machine Learning

This project implements a complete end-to-end pipeline for detecting exoplanets in Kepler Space Telescope data. It combines two complementary approaches:
1.  **Deep Learning (CNN):** A Convolutional Neural Network trained on phase-folded light curves to classify transit shapes. - Chikara Oe, Iori Adachi
2.  **Bayesian Analysis:** A probabilistic frequency analysis to recover orbital periods from raw time-series data. - Linus Upson

---

## Quick Start Guide

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/chikaranabla/finding_exoplanet-project3.git](https://github.com/chikaranabla/finding_exoplanet-project3.git)
cd finding_exoplanet-project3
pip install -r requirements.txt
```

### 2. Get the Data
There are different options: Download (recommended because it takes hours to download and clean) or generate it.

#### Option A: Download Pre-Processed Data from Our Script
1. [Click here to download the data zip file (Google Drive)](https://drive.google.com/file/d/1BCEJuv4PPvfDR1Ro-EoRoTfK9_bY5g5e/view?usp=sharing)
2. Unzip file.
3. Place `pipeline_output_final_data` folder in the root directory of this project.

#### Option B: Generate the Data from Scratch
```bash
python 1_generate_data.ipynb
```
Input: `q1_q8_koi_2025.02.03_04.12.15.csv` (Kepler Object of Interest Catalog)
Output: Creates the `pipeline_output_final_data/` folder containing .npz batches and metadata.
Time: ~2-3 hours, depending on internet connection.

### 3. Train the CNN
Train the Deep Learning model to distinguish between Planets and False Positives.
```bash
python 2_train_cnn.py
```
Output: Training logs (Accuracy/Loss), Confusion Matrix plot, and Filter visualizations.
Current Performance: ~85% Validation Accuracy.

### 4. Run Bayesian Analysis
Perform frequency analysis on specific KOI object light curves to recover their orbital periods.
```bash
python 3_run_bayesian.py
```
Output: Data and figures showing the most likely orbital period and phase-folded comparisons.

---

# Project Architecture

`1_generate_data.ipynb` (Data Loading/Preprocessing)
- Dual-View Generation: Creates "Global Views" (full orbit, 2001 bins) and "Local Views" (transit zoom, 201 bins) for the CNN.
- Random Folding: Handles False Positives that lack orbital periods by assigning random folding parameters, ensuring the CNN learns to reject non-periodic noise.
- Augmentation: Uses the batman package to inject synthetic transits into quiet stars to balance the dataset.

`2_train_cnn.py` (ML Model)
- Architecture: A dual-input 1D CNN.
    - Local Column: High-resolution view of the transit shape.
    - Global Column: Low-resolution view of the full light curve.

`3_run_bayesian.py` (Bayesian Analysis)
- Runs Bayesian periodicity analysis on the light curves.
- Uses functions from `BayesianSignal.py`.

`BayesianSignal.py` (Functions For Bayesian Analysis)
- Functions include:
    - Calculating the posterior probability of a periodic signal vs. a constant noise model and a non-periodic model.
    - Marginalizing over phase and model bins to estimate the orbital frequency ($\omega$).
    - Experimental models, phase-folding, other helpers.


