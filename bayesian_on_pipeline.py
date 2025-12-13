#%%

import numpy as np
import pandas as pd
import os
from BayesianSignal import *


light_curve_dir = 'pipeline_output_final/bayesian_unfolded/'
light_curve_files = os.listdir(light_curve_dir)
kic_available = [
    int(fname.split('_')[1]) 
    for fname in light_curve_files 
    if fname.startswith('kic_') and fname.endswith('_unfolded.csv')
]

# kics = [1871056,2584163, ]
# kic = kics[0]

# lc = pd.read_csv(os.path.join(light_curve_dir, f'kic_{kic}_unfolded.csv'))
# for kic in kic_available:
#     # print(f"Processing KIC {kic}")
#     lc = pd.read_csv(os.path.join(light_curve_dir, f'kic_{kic}_unfolded.csv'))
#     t = lc['time'].values
#     signal = lc['flux'].values
#     signal_err = lc['flux_err'].values

#     meta = pd.read_csv('q1_q8_koi_2025.02.03_04.12.15.csv', header=53)
#     row = meta[meta['kepid']==kic].iloc[0]

#     period = float(row['koi_period'])
#     t0 = float(row['koi_time0bk'])
#     transit_duration = float(row['koi_duration'])/24 #days
#     omega_true = 2 * np.pi / period

#     print("KIC:", kic, (t.max()-t.min())/period, "Periods in data span")

kic = 4249725 #6103377 #
lc = pd.read_csv(os.path.join(light_curve_dir, f'kic_{kic}_unfolded.csv'))
t = lc['time'].values
signal = lc['flux'].values
signal_err = lc['flux_err'].values

signal, signal_err = normalize_signal(signal, signal_err)

meta = pd.read_csv('q1_q8_koi_2025.02.03_04.12.15.csv', header=53)
row = meta[meta['kepid']==kic].iloc[0]

period = float(row['koi_period'])
t0 = float(row['koi_time0bk'])
transit_duration = float(row['koi_duration'])/24 #days
omega_true = 2 * np.pi / period

total_time = t.max()-t.min()

# %%
omega_min = omega_true /3#2*np.pi/((np.max(t)-np.min(t))/10)
omega_max = omega_true *3
# max_spacing = 1/(2*np.pi * time_total) #* Largest space between omega values to ensure no peaks are missed
def optimal_spaced_omega(omega_min, omega_max, total_time, transit_duration):
    """Calculate optimal spacing of omega values based on transit duration and total observation time."""
    N = int(np.ceil(np.log(omega_max/omega_min) / np.log(1+ transit_duration/total_time)))
    n = np.arange(0, N+1)
    return omega_min * np.pow(1 + transit_duration/total_time, n)
# omega_spacing = omega_true * transit_duration/total_time
# num_omega = int((omega_max-omega_min)/omega_spacing)#math.ceil((omega_max - omega_min) / max_spacing)
# # print(f"Sampling {num_omega} omega values between {omega_min} and {omega_max}")
# Omega = np.linspace(omega_min, omega_max, num_omega)

Omega = optimal_spaced_omega(omega_min, omega_max, total_time, transit_duration)
num_omega = len(Omega)

# Omega = np.sort(np.concatenate((Omega, [omega_true,])))

Phi = np.linspace(0, 2*np.pi, 3)

M = np.arange(60, 240, 20)

print(f"KIC {kic} has period {period} days, frequency {omega_true} rad/day")
print(f"Loading light curve with {len(lc)} data points")
print(f"Calculating Omega curve with {num_omega} omega values between {omega_min} and {omega_max}")
print(f"Using {len(Phi)} phi values and {len(M)} m values (between {M.min()} and {M.max()})")

r_mean = np.mean(signal)
r_min = np.min(signal-3*signal_err)
r_max = np.max(signal+3*signal_err)
log_P_D = calculate_logprob_fast(t, signal, signal_err, Omega, Phi, M, r_min=r_min, r_max=r_max)
P_w, log_offset = marginalize_Phi_M(log_P_D, Omega, Phi, M, omega_min, omega_max)

#%%
log_odds_ratio(log_P_D, log_P_constant(t, signal, signal_err, A_min=r_min, A_max=r_max), Omega, Phi, M, omega_min, omega_max)

# %%
fig, axs = plt.subplots(1, 4, figsize=(13, 4))

omega_best = best_omega(Omega, P_w)
period_percent_err = np.abs((2*np.pi/omega_best - period)/period)*100
print("Period Ratio:", omega_best/omega_true, "Best Period:", 2*np.pi/omega_best, "True Period:", period, "Percent Error:", period_percent_err)
num_bins_display = 400
folded_signal, bin_phases = phase_fold(t, signal, omega=omega_best, phi=0, num_bins=num_bins_display, reduction='mean')

# First subplot: the signal versus time
axs[0].plot(t, signal, lw=0.2)
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Signal')
axs[0].set_title('Observed Signal')

# Second subplot: Omega vs. Pw (current plot)
axs[1].plot(2*np.pi/Omega, P_w)
axs[1].axvline(x=2*np.pi/omega_true, color='r', linestyle='--', label='True Frequency')
axs[1].axvline(x=2*np.pi/omega_best, color='g', linestyle='--', label='Best Fit Frequency')
axs[1].set_xlabel('Period (days)')#(r'Frequency $\omega$')
axs[1].set_ylabel(r"Probability $P(\omega | D, I)$")
axs[1].legend()
axs[1].set_title('Probability vs. Frequency')
axs[1].ticklabel_format(axis='y', style='sci')

# Third subplot: Phase-folded signal using best-fit frequency
axs[2].plot(bin_phases, folded_signal)
axs[2].set_xlabel('Phase')
axs[2].set_ylabel('Folded Signal')
axs[2].set_title(rf'Phase-Folded Signal ($\omega={omega_best:.2f}$)')
axs[2].axhline(y=r_min, color='r', linestyle='--', label='Min Signal Level')
axs[2].axhline(y=r_max, color='g', linestyle='--', label='Max Signal Level')

folded_signal_true, bin_phases_true = phase_fold(t, signal, omega=omega_true, phi=0, num_bins=num_bins_display, reduction='mean')
axs[3].plot(bin_phases_true, folded_signal_true)
axs[3].set_xlabel('Phase')
axs[3].set_ylabel('Folded Signal')
axs[3].set_title(rf'True Folded Signal ($\omega={omega_true:.2f}$)')
axs[3].axhline(y=r_min, color='r', linestyle='--', label='Min Signal Level')
axs[3].axhline(y=r_max, color='g', linestyle='--', label='Max Signal Level')

axs[3].sharey(axs[2])

plt.tight_layout()
plt.show()
# %%
row
# %%
omega_peak_right =omega_best + 2e-5
omega_peak_left =omega_best - 2e-5

Omega_peak = np.linspace(omega_peak_left, omega_peak_right, 200)

log_P_D_peak = calculate_logprob(t, signal, signal_err, Omega_peak, Phi, M, r_min=r_min, r_max=r_max)
P_w_peak, log_offset_peak = marginalize_Phi_M(log_P_D_peak, Omega_peak, Phi, M, omega_min, omega_max)
# peak_mask = np.logical_and(, Omega > omega_best - 0.0005)
# peak_Omega, peak_P = Omega[peak_mask], P_w[peak_mask]
plt.plot(Omega_peak, P_w_peak, label='Zoomed Omega Curve')
# plt.yscale('log')
# plt.ylim(bottom=1e-4)
plt.show()
# %%
