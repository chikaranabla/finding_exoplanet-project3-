# Exoplanet Data Pipeline (Kepler)

**Authors:** Zack
**Branch:** Zack_Data_Cleaning
**Project:** PHYS 188 Project III - Exoplanet Detection using Deep Learning

## Overview
This pipeline downloads Kepler light curves, removes noise (outliers/stellar variability), and formats them for our models.

It produces two distinct outputs:
1. For ML Team (Chikara, Iori, Levi): A normalized, fixed-length tensor of light curves suitable for CNN input.
1. For Bayesian Team (Linus): A metadata key containing Kepler IDs (KIC) and true orbital periods to facilitate physics-based signal detection.

## How to use this code
1. **Install requirements:**
   `pip install -r requirements.txt`
2. **Place the Input CSV:**
   Ensure `q1_q8_koi_2025.02.03_04.12.15.csv` is in the root folder
   Run `python full_pipeline.py`.
   * This will download the stars, process them, and save them into a folder called `processed_dataa_new/`.
   * **Note:** It creates checkpoints. If it stops, run it again and it resumes.
3. **Check the Data:**
   Run `python inspect_data.py` to see a plot of a planet.

## Output Files (Generated Locally)
* `final_dataset/X_data.npy`: Input features for CNN (Shape: N x 2000; Stitched, Flattened, Interpolated).
* `final_dataset/y_labels.npy`: Labels (1=Planet, 0=False Positive).
* `final_dataset/dataset_key.csv`: Metadata for Bayesian analysis (kic: Kepler Input Catalog ID, label: Class Name, period: Orbital Period (days), duration: Transit Duration (hours)).

## Technical Details
* Source: NASA Exoplanet Archive (MAST) via `lightkurve`.
* Resolution: Default is set to download Quarter 1 only for speed (`USE_SINGLE_QUARTER = True` in script). Toggle to `False` to download full 4-year missions (warning: takes hours).
* Parallelism: Uses `ThreadPoolExecutor` with 4 workers.
