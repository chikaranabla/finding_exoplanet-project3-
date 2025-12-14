# Exoplanet Data Pipeline (Kepler)

**Authors:** Zack
**Branch:** Zack_Data_Cleaning
**Project:** PHYS 188 Project III - Exoplanet Detection using Deep Learning

## Overview
This pipeline downloads Kepler light curves, removes noise (outliers/stellar variability), and formats them for our models.

It produces two distinct outputs:
1. For ML Team (Chikara, Iori, Levi): A normalized, fixed-length tensor of light curves suitable for CNN input.
1. For Bayesian Team (Linus): A metadata key containing Kepler IDs (KIC) and true orbital periods to facilitate physics-based signal detection.

## 📥 Download the Dataset
The dataset is too large for GitHub, so it is hosted externally.
**[Click here to download kepler_dataset_v1.zip via Google Drive](https://drive.google.com/file/d/1BCEJuv4PPvfDR1Ro-EoRoTfK9_bY5g5e/view?usp=sharing)**

After downloading:
1. Unzip the file.
2. Place the `pipeline_output_final_data` folder in the same directory as these notebooks.



## How to use this code
1. **Install requirements:**
   `pip install -r requirements.txt`
2. **Place the Input CSV:**
   Ensure `q1_q8_koi_2025.02.03_04.12.15.csv` is in the root folder.

For the CNN Team:
Do not load individual CSVs. Use the batched .npz files for high-speed loading.

import numpy as np
import pandas as pd

data = np.load('batch_1.npz')
global_view = data['flux_global']
local_view  = data['flux_local']
labels      = data['label']
print(f"Loaded {len(labels)} samples.")


For Bayesian Team:
Use the bayesian_unfolded folder. You need the full time series to search for periodicity.

import pandas as pd

df = pd.read_csv('bayesian_unfolded/kic_3733346_unfolded.csv')
time = df['time']
flux = df['flux']

Stage 1: Core Processing (Download & Clean)
Source: Official Kepler data accessed via the lightkurve API.

Stitching: All available quarters were stitched together to remove instrumental offsets.

Detrending: We applied a flattening filter (Savitzky-Golay, window=101) to remove stellar variability (spots/rotation) while preserving the sharp transit signals.

Normalization: Flux is normalized to Median=0, Min=-1.

Stage 2: Data Augmentation & Balancing
The Problem: The raw catalog is imbalanced (mostly confirmed planets) and lacks "hard" negative examples.

The Solution: We injected synthetic transits (using the batman package) into quiet stars to create more positive samples.

Current Balance: The dataset includes a mix of Real Planets (Label 1), Real False Positives (Label 0), and Synthetic Planets (Label 1).

Stage 3: Dual-View Generation (For CNN)
Global View (2001 bins): The light curve is phase-folded on the orbital period to show the full periodic shape.

Local View (201 bins): The light curve is zoomed in on the primary transit event to capture detailed ingress/egress shapes.


Key Decisions
1. Switching to Lightkurve: moved away from manual FITS file handling to the lightkurve library. This standardized the "stitching" process, ensuring that data from different telescope rotations aligned perfectly.
2. Random Folding for False Positives: Many "False Positive" stars in the catalog lack orbital periods (because they aren't periodic). Instead of discarding them, we assigned random folding parameters to these stars. This teaches the CNN that "random noise folded on a random period = Label 0," which is crucial for reducing false alarms.
3. Handling "Empty" Local Views: Originally, zooming in on a specific time ($T_0$) often resulted in empty data if the telescope was off during that specific transit. The updated pipeline folds on the actual period for the local view as well. This stacks every transit on top of each other, filling in the gaps.

Known Challenges Handled:
1. Network Instability: The pipeline includes robust error handling for truncated downloads (NASA MAST server timeouts). Failed downloads were automatically skipped and logged, ensuring the final dataset contains only valid, non-corrupted files.
2. Leap Second Warnings: You may see ErfaWarning: dubious year when processing. These are harmless artifacts of the time conversion library and do not affect the flux values or model training.

