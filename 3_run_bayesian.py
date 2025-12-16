#%%
#* This file runs the Bayesian periodicity detection algorithm on light curves provided by the data processing pipeline. The functions it uses are found in BayesianSignal.py

import numpy as np
import pandas as pd
import os
from BayesianSignal import *


light_curve_dir = 'pipeline_output_final/bayesian_unfolded/' #! CHANGE PATH TO LIGHT CURVE FILES OUTPUT FROM DATA PROCESSING PIPELINE
light_curve_files = os.listdir(light_curve_dir)
kic_available = [
    int(float(fname.split('_')[1]))
    for fname in light_curve_files 
    if fname.startswith('kic_') and fname.endswith('_unfolded.csv')
]

output_dir = 'bayesian_output_mod/' #! Where to save output files (plots, results CSV, calculated P(D|w, phi, M_m,I) arrays, etc)

period_min = 2
period_max = 100

omega_min = 2*np.pi/period_max
omega_max = 2*np.pi/period_min

transit_duration = 3/24 #choose rough ballpark of minimum transit duration (3 hrs)


meta = pd.read_csv('q1_q8_koi_2025.02.03_04.12.15.csv', header=53) #! CHANGE PATH TO METADATA FILE FROM KAGGLE DATASET
meta_loaded = pd.read_csv('metadata_final.csv').set_index('kic') #! CHANGE PATH TO METADATA FILE OUTPUT FROM DATA PROCESSING PIPELINE

for kic in kic_available:
    results_filename = os.path.join(output_dir, "bayesian_pipeline_results.csv")
    if os.path.exists(results_filename):
        results_df = pd.read_csv(results_filename)
    else:
        results_df = pd.DataFrame(columns=['kic', 'omega_best', 'log_lr_constant', 'log_lr_nonperiodic'])

    if kic in results_df['kic'].values:
        print(f"KIC {kic} already processed, skipping.")
        continue

    print(f"Processing KIC {kic} ----------------------------------------------------------------------------------")
    try:
        try:
            lc = pd.read_csv(os.path.join(light_curve_dir, f'kic_{kic}_unfolded.csv'))
        except FileNotFoundError:
            lc = pd.read_csv(os.path.join(light_curve_dir, f'kic_{kic}.0_unfolded.csv'))
    except pd.errors.ParserError:
        print(f"Error reading light curve for KIC {kic}, skipping.")
        continue

    t = lc['time'].values
    signal = lc['flux'].values
    signal_err = lc['flux_err'].values

    # Check for NaNs in t, signal, or signal_err; skip this kic if any are found
    if np.any(np.isnan(t)) or np.any(np.isnan(signal)) or np.any(np.isnan(signal_err)) or np.any(signal_err <= 0.0):
        print(f"KIC {kic} contains NaN values in time/flux/flux_err arrays, skipping.")
        continue
    elif kic in meta_loaded.index and np.any(meta_loaded.loc[kic, 'is_synthetic']):
        print(f"KIC {kic} is synthetic, skipping.")
        continue

    signal, signal_err = normalize_signal(signal, signal_err)
    r_mean = np.mean(signal)
    r_min = np.min(signal-3*signal_err)
    r_max = np.max(signal+3*signal_err)

    total_time = t.max()-t.min()

    #* Load true values for comparison/plotting
    row = meta[meta['kepid']==kic].iloc[0]
    period = float(row['koi_period'])
    t0 = float(row['koi_time0bk'])
    omega_true = 2 * np.pi / period
    
    #* Generate grid of Omega, Phi, M values
    Omega = optimal_spaced_omega(omega_min, omega_max, total_time, transit_duration)
    num_omega = len(Omega)
    Phi = np.array([0.0, np.pi])#np.linspace(0, 2*np.pi, 2)
    M = np.array([100, 200])#np.arange(60, 240, 20)

    print(f"KIC {kic} has period {period} days, frequency {omega_true} rad/day")
    print(f"Loading light curve with {len(lc)} data points")
    print(f"Calculating Omega curve with {num_omega} omega values between {omega_min} and {omega_max}")
    print(f"Using {len(Phi)} phi values and {len(M)} m values (between {M.min()} and {M.max()})")


    starttime = time.perf_counter()
    log_P_D = calculate_logprob_numba(t, signal, signal_err, Omega, Phi, M, r_min=r_min, r_max=r_max)
    exectime = time.perf_counter() - starttime
    print(f"Calculated log probabilities in {exectime:.2f} seconds")
    P_w, log_offset = marginalize_Phi_M(log_P_D, Omega, Phi, M, omega_min, omega_max)

    omega_best = best_omega(Omega, P_w)

    log_lr_constant = log_odds_ratio_constant(log_P_D, log_P_constant(t, signal, signal_err, A_min=r_min, A_max=r_max), Omega, Phi, M, omega_min, omega_max)

    P_vals_nonperiodic, log_offset_nonperiodic = P_nonperiodic(t, signal, signal_err, Phi, M, r_min=0, r_max=1)
    log_lr_nonperiodic = log_odds_ratio_nonperiodic(log_P_D, P_vals_nonperiodic, log_offset_nonperiodic, Omega, Phi, M, omega_min, omega_max)

    print("Log Odds Ratio (Periodic Model / Constant Model):", log_lr_constant, 
        "\nLog Odds Ratio (Periodic Model / Non-Periodic Model):", log_lr_nonperiodic)
    
    #* Save analysis results to CSV
    out_results = {'kic': kic, 'omega_best': omega_best, 'log_lr_constant': log_lr_constant, 'log_lr_nonperiodic': log_lr_nonperiodic}

    # Append the current results as a new row
    results_df = pd.concat([results_df, pd.DataFrame([out_results])], ignore_index=True)
    # Overwrite the file with updated results
    results_df.to_csv(results_filename, index=False)

    #* Save log_P_D
    output_curve_dir = os.path.join(output_dir, 'curves')
    os.makedirs(output_curve_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_curve_dir, f'kic_{kic}_log_P_D.npz'), Omega=Omega, Phi=Phi, M=M, log_P_D=log_P_D)

    #* Plotting Section --------------------------------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 5, figsize=(15, 4))

    period_percent_err = np.abs((2*np.pi/omega_best - period)/period)*100
    print("Period Ratio:", omega_best/omega_true, "Best Period:", 2*np.pi/omega_best, "True Period:", period, "Percent Error:", period_percent_err)
    num_bins_display = 400
    folded_signal, bin_phases = phase_fold(t, signal, omega=omega_best, phi=0, num_bins=num_bins_display, reduction='mean')

    #* First subplot: the signal versus time
    axs[0].plot(t, signal, lw=0.2)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Normalized Signal')
    axs[0].set_title('Observed Light Curve')

    #* Second subplot: Period vs. P(w|D,I)
    axs[1].plot(2*np.pi/Omega, P_w)
    axs[1].axvline(x=2*np.pi/omega_true, color='r', linestyle='--', label='True Period')
    axs[1].axvline(x=2*np.pi/omega_best, color='g', linestyle='--', label='Best Fit Period')
    axs[1].set_xlabel('Period (days)')
    axs[1].set_ylabel(r"Probability $P(\omega | D, I)$")
    axs[1].set_xscale('log')
    axs[1].legend()
    axs[1].set_title('Probability vs. Period')
    axs[1].ticklabel_format(axis='y', style='sci')

    #* Third subplot: Phase-folded signal using best-fit frequency
    axs[2].plot(bin_phases, folded_signal)
    axs[2].set_xlabel('Phase')
    axs[2].set_ylabel('Folded Signal')
    axs[2].set_title(rf'Phase-Folded Signal ($\omega={omega_best:.2f}$)')
    axs[2].axhline(y=r_min, color='r', linestyle='--', label='Min Signal Level')
    axs[2].axhline(y=r_max, color='g', linestyle='--', label='Max Signal Level')
    axs[2].legend()

    #* Fourth subplot: Phase-folded signal using true frequency
    folded_signal_true, bin_phases_true = phase_fold(t, signal, omega=omega_true, phi=0, num_bins=num_bins_display, reduction='mean')
    axs[3].plot(bin_phases_true, folded_signal_true)
    axs[3].set_xlabel('Phase')
    axs[3].set_ylabel('Folded Signal')
    axs[3].set_title(rf'True Folded Signal ($\omega={omega_true:.2f}$)')
    axs[3].axhline(y=r_min, color='r', linestyle='--', label='Min Signal Level')
    axs[3].axhline(y=r_max, color='g', linestyle='--', label='Max Signal Level')
    axs[3].legend()
    axs[3].sharey(axs[2])

    #* Fifth subplot: Zoomed-in Omega curve around true frequency
    peak_width = omega_best * transit_duration/total_time
    omega_peak_right =omega_best + peak_width
    omega_peak_left =omega_best - peak_width

    Omega_peak = np.linspace(omega_peak_left, omega_peak_right, 200)

    #* Recalculate log_P_D for the zoomed-in Omega range
    log_P_D_peak = calculate_logprob(t, signal, signal_err, Omega_peak, Phi, M, r_min=r_min, r_max=r_max)
    P_w_peak, log_offset_peak = marginalize_Phi_M(log_P_D_peak, Omega_peak, Phi, M, omega_min, omega_max)
    Period_peak = 2*np.pi/Omega_peak
    zoomed_peak_period = Period_peak[np.argmax(P_w_peak)]
    min_conv = 24 * 60
    axs[4].plot((Period_peak-period)*min_conv, P_w_peak)
    axs[4].set_xlabel(r'Deviation from True Period (min)')
    axs[4].set_ylabel(r'Probability $P(\omega | D, I)$ (Arbitrary Scaling)')
    axs[4].axvline(x=0, color='r', linestyle='--', label='True Period')
    axs[4].axvline(x=(zoomed_peak_period-period)*min_conv, color='g', linestyle='--', label='Best-Fit Period')
    axs[4].set_title(rf'$P(\omega | D, I)$ Around True Period')
    axs[4].legend()
    axs[4].set_yscale('log')

    plt.suptitle("Bayesian Periodicity Analysis for KIC {}".format(kic))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'kic_{kic}_bayesian_analysis.png'))
    plt.close()
