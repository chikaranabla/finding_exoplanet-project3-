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
   Run `python pipeline_test.py`.
   * This will download the stars, process them, and save them into a folder called `pipeline_output_v1/`.
   * **Note:**
      * It creates checkpoints. The notebook may hang after several stars. If it stops, restart the kernel and re-run the pipeline (it resumes).
      * To create more rows, check my comments in the `run_pipeline` function where I change the parameters from 300->1000 and 300->. The synthetic data will help because there aren't many planets with 50-day periods.
      * *Pre-Filtering*: Only downloads planets >50 days and <4 Earth Radii (saving you time).
      * *Anti-Hanging Protection*: Added a `timeout=60s` and a "Single Quarter Fallback" so it doesn't get stuck downloading.
      * It may be helpful to download the `metadata_final.csv` from my branch because my pipeline is designed to pickup where it left off... Hopefully this works so you can save time making the data.
3. **Check the Data:**
   The last cell creates visuals.

## Output Files (Generated Locally)
* `X_data.npy`: Input features for CNN (Shape: N x 2000; Stitched, Flattened, Interpolated).
* `metadata_final.csv`: Metadata for Bayesian analysis (kic: Kepler Input Catalog ID, label: Class Name, period: Orbital Period (days), duration: Transit Duration (hours)).

## Technical Details
* Source: NASA Exoplanet Archive (MAST) via `lightkurve`.
* Resolution: Default is set to download Quarter 1 only for speed (`USE_SINGLE_QUARTER = True` in script). Toggle to `False` to download full 4-year missions (warning: takes hours).
* Parallelism: Uses `ThreadPoolExecutor` with 4 workers.



