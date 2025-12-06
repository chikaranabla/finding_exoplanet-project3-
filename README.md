# Modified Data Cleaning Pipeline
Original authors: Zack and Isabella 
Modification by Iori

## Overview (Copied from Combined_Data_Cleaning branch)
This pipeline downloads Kepler light curves, removes noise (outliers/stellar variability), and formats them for our models.

It produces two distinct outputs:
1. For ML Team (Chikara, Iori, Levi): A normalized, fixed-length tensor of light curves suitable for CNN input.
1. For Bayesian Team (Linus): A metadata key containing Kepler IDs (KIC) and true orbital periods to facilitate physics-based signal detection.

## Modifications
The ouput for the ML process now produces two files X_Global (2001 points) and X_Local (201 points) based on the paper by Shallue and Vanderburg.
I added a helper function that generates the local view.
Also decreased BLS grid values just to make it faster during debugging, feel free to change it back. 
* I think the original value is
* durations = np.linspace(0.05, 0.5, 10)
* period_grid = np.linspace(1, 400, 5000)
Added a .py version so you can run it from terminal

