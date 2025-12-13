import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import simpson
from scipy.signal import find_peaks
from scipy.special import loggamma, erfc
import progressbar
import concurrent.futures
import os

def phase_fold(t, signal, omega, phi, num_bins, reduction='sum'):
    phase = ((t * omega + phi) / (2 * np.pi)) % 1
    timestamp_bins = np.floor(phase * num_bins).astype(int)

    bin_signal = np.bincount(timestamp_bins, weights=signal)

    timestamps_per_bin = np.bincount(timestamp_bins)

    # if np.any(timestamps_per_bin <= 0):
    #     raise ValueError(f"Zero timestamps_per_bin encountered: omega={omega}, phi={phi}, num_bins={num_bins}")

    # bin_average = bin_signal / timestamps_per_bin

    bin_phases = np.linspace(0, 1, num_bins)

    if reduction == 'mean':
        bin_signal = bin_signal / timestamps_per_bin
    elif reduction == 'sum':
        pass
    else:
        raise ValueError("reduction must be either 'sum' or 'mean'")

    return bin_signal, bin_phases

def normalize_signal(signal, uncerts=None):
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    if uncerts is None:
        return (signal - signal_min) / (signal_max - signal_min) 
    else:
        norm_signal = (signal - signal_min) / (signal_max - signal_min)
        norm_uncerts = uncerts / (signal_max - signal_min)
        return norm_signal, norm_uncerts

def make_dummy_signal(time_total, n_timesteps, omega=None, period=None, noise_ratio=0.1, signal_type="sin"):

    if omega is None and period is None:
        raise ValueError("Either omega or period must be provided")
    if omega is not None and period is not None:
        raise ValueError("Only one of omega or period must be provided")
    if omega is None:
        omega = 2*np.pi/period
    
    t = np.linspace(0, time_total, n_timesteps)
    if signal_type == "sin":
        signal = 0.5+ 0.5*np.sin(omega * t)
    elif signal_type == "square":
        signal = 0.5+ 0.5*np.sign(np.sin(omega * t)+0.6)
    return t, normalize_signal(signal + np.random.normal(0, noise_ratio, n_timesteps))


def best_omega(Omega, Pw):
    return Omega[Pw.argmax()]


def bin_times(times, omega, phi, m):
    return np.floor(m * ((times * omega + phi) % (2 * np.pi)) / (2 * np.pi)).astype(int)

#* m is always num_bins

def bin_sum(values, bins, m):
    return np.bincount(bins, weights=values, minlength=m)

# Using algorithm from https://iopscience.iop.org/article/10.1086/307433/pdf
def calculate_logprob(t, signal, uncerts, Omega, Phi, M, r_min, r_max):
    """Calculates the log prob P(D | w, b=1, phi, M_m, I) on a grid specified by Omega, Phi, M

    Args:
        t (np.array): The times the signal is sampled at
        signal (np.array): The signal (as a flux) observed at times t
        uncerts (np.array): Uncertainties on the signal observations at times t
        Omega (np.array): The omega values to evaluate the log prob at
        Phi (np.array): The phi values to evaluate the log prob at
        M (np.array): The m values (number of bins) to evaluate the log prob at
        r_min (float): The minumum possible 'true flux' value (used for priors on the flux)
        r_max (float): The maximum possible 'true flux' value (used for priors on the flux)

    Raises:
        ValueError: If any bins end up with 0 examples in them

    Returns:
        np.array(shape=(len(Omega), len(Phi), len(M))): The log prob sampled on the specified Omega, Phi, M grid
    """
    delta_r = r_max - r_min
    N = len(t)

    log_P_D = np.zeros((len(Omega), len(Phi), len(M)))
    bar = progressbar.ProgressBar(maxval=len(Omega))
    for i, w in bar(enumerate(Omega)):
        for j, p in enumerate(Phi):
            for k, m in enumerate(M):

                bins = bin_times(t, w, p, m)

                w_j = np.bincount(bins, minlength=m) > 1 #* see appendix of paper, deals with when bins have < 2 samples
                
                W_j = bin_sum(1/np.square(uncerts), bins, m)

                W_j[w_j==0] = 1.0 #* Modification from appendix, set to 1

                # if np.any(W_j == 0):
                #     raise ValueError(f"Zero W_j encountered: omega={w}, phi={p}, m={m}")
                
                d_Wj = bin_sum(signal/np.square(uncerts), bins, m)/W_j

                d_Wj2 = bin_sum(np.square(signal/uncerts), bins, m)/W_j

                chisq_Wj = W_j * (d_Wj2 - np.square(d_Wj)) * w_j / (1/m * np.sum(w_j)) #* modification to chisq_Wj from (A1) in appendix

                y_jmin = np.sqrt(0.5 * W_j) * (r_min - d_Wj)
                y_jmax = np.sqrt(0.5 * W_j) * (r_max - d_Wj)

                erfc_terms = 1/np.sqrt(W_j) * (erfc(y_jmin) - erfc(y_jmax))

                log_P_D[i, j, k] = (-0.5*N*np.log(2*np.pi)) + (-m * np.log(delta_r)) - np.sum(np.log(uncerts)) +\
                (0.5 * m * np.log(np.pi/2)) - np.sum(chisq_Wj/2) + np.sum(np.log(erfc_terms))
                
    return log_P_D # This is ln(P(D|w, b=1, phi, M_m, I)) as defined in eqn 24


def calculate_logprob_fast(t, signal, uncerts, Omega, Phi, M, r_min, r_max, n_jobs=None, chunk_size=None, show_progress=True):
    """Calculates the log prob P(D | w, b=1, phi, M_m, I) on a grid specified by Omega, Phi, M

    Args:
        t (np.array): The times the signal is sampled at
        signal (np.array): The signal (as a flux) observed at times t
        uncerts (np.array): Uncertainties on the signal observations at times t
        Omega (np.array): The omega values to evaluate the log prob at
        Phi (np.array): The phi values to evaluate the log prob at
        M (np.array): The m values (number of bins) to evaluate the log prob at
        r_min (float): The minumum possible 'true flux' value (used for priors on the flux)
        r_max (float): The maximum possible 'true flux' value (used for priors on the flux)
        n_jobs (int | None): Number of parallel worker threads across Omega. Defaults to min(cpu_count, len(Omega)).
        chunk_size (int | None): Number of Omega entries per task. Defaults to about 4 chunks per worker.
        show_progress (bool): Whether to display a progress bar across Omega chunks.

    Raises:
        ValueError: If any bins end up with 0 examples in them

    Returns:
        np.array(shape=(len(Omega), len(Phi), len(M))): The log prob sampled on the specified Omega, Phi, M grid
    """
    delta_r = r_max - r_min
    N = len(t)

    one_over_square_uncerts = 1/np.square(uncerts)

    d_Wj_unbinsummed = signal/np.square(uncerts)

    d_Wj2_unbinsummed = np.square(signal/uncerts)

    sum_log_uncerts = np.sum(np.log(uncerts))

    num_omega = len(Omega)
    num_phi = len(Phi)
    num_m = len(M)

    log_P_D = np.zeros((num_omega, num_phi, num_m))

    # Determine workers and chunking
    if n_jobs is None or n_jobs <= 0:
        cpu = os.cpu_count() or 1
        n_jobs = min(cpu, num_omega)
    else:
        n_jobs = min(n_jobs, num_omega)

    if chunk_size is None or chunk_size <= 0:
        target_chunks = max(1, n_jobs * 4)
        chunk_size = max(1, (num_omega + target_chunks - 1) // target_chunks)
    print("PARALELL EXECUTION: Using n_jobs =", n_jobs, "with chunk_size =", chunk_size)

    def worker(start_idx, end_idx):
        chunk_len = end_idx - start_idx
        out = np.zeros((chunk_len, num_phi, num_m))
        for local_i, w in enumerate(Omega[start_idx:end_idx]):
            for j, p in enumerate(Phi):
                for k, m in enumerate(M):
                    bins = bin_times(t, w, p, m)

                    w_j = np.bincount(bins, minlength=m) > 1 #* see appendix of paper, deals with when bins have < 2 samples

                    W_j = bin_sum(one_over_square_uncerts, bins, m)

                    W_j[w_j==0] = 1.0 #* Modification from appendix, set to 1

                    # if np.any(W_j == 0):
                    #     raise ValueError(f"Zero W_j encountered: omega={w}, phi={p}, m={m}")
                    d_Wj = bin_sum(d_Wj_unbinsummed, bins, m) / W_j
                    d_Wj2 = bin_sum(d_Wj2_unbinsummed, bins, m) / W_j
                    
                    chisq_Wj = W_j * (d_Wj2 - np.square(d_Wj)) * w_j / (1/m * np.sum(w_j)) #* modification to chisq_Wj from (A1) in appendix

                    y_jmin = np.sqrt(0.5 * W_j) * (r_min - d_Wj)
                    y_jmax = np.sqrt(0.5 * W_j) * (r_max - d_Wj)
                    erfc_terms = 1/np.sqrt(W_j) * (erfc(y_jmin) - erfc(y_jmax))
                    out[local_i, j, k] = (-0.5*N*np.log(2*np.pi)) + (-m * np.log(delta_r)) - sum_log_uncerts +\
                        (0.5 * m * np.log(np.pi/2)) - np.sum(chisq_Wj/2) + np.sum(np.log(erfc_terms))
        return start_idx, out

    # Parallel execution using threads
    tasks = []
    bar = None
    completed = 0
    if show_progress and num_omega > 0:
        bar = progressbar.ProgressBar(maxval=num_omega)
        bar.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for start in range(0, num_omega, chunk_size):
            end = min(num_omega, start + chunk_size)
            tasks.append(executor.submit(worker, start, end))
        for fut in concurrent.futures.as_completed(tasks):
            start_idx, out = fut.result()
            end_idx = start_idx + out.shape[0]
            log_P_D[start_idx:end_idx, :, :] = out
            if bar is not None:
                completed += out.shape[0]
                bar.update(min(completed, num_omega))
    if bar is not None:
        bar.finish()

    return log_P_D # This is ln(P(D|w, b=1, phi, M_m, I)) as defined in eqn 24

def log_P_constant(t, signal, uncerts, A_min, A_max):
    """Calculate the log_prob P(D | b=1, I) for a constant signal model (equivalent to m=1 case)

    Args:
        t (np.array): The times the signal is sampled at
        signal (np.array): The signal (as a flux) observed at times t
        uncerts (np.array): The uncertainties on the signal observations at times t
        A_min (float): The minimum possible 'true flux' value
        A_max (float): The maximum possible 'true flux' value

    Returns:
        float: The log prob ln(P(D | b=1, I))
    """
    # Replace all b*W_j with W_j and s_i*b^-1/2 with s_i, with the exception that d_Wj and d_Wj2 remain unchanged
    N = len(t)
    delta_A = A_max - A_min

    W = np.sum(1/np.square(uncerts))

    d_W = np.sum(signal/np.square(uncerts))/W

    d_W2 = np.sum(np.square(signal/uncerts))/W

    chisq_W = W * (d_W2 - np.square(d_W))

    y_Amin = np.sqrt(0.5 * W) * (A_min - d_W)
    y_Amax = np.sqrt(0.5 * W) * (A_max - d_W)

    erfc_term = 1/np.sqrt(W) * (erfc(y_Amin) - erfc(y_Amax))

    log_P_D = (-0.5 * N * np.log(2*np.pi)) - np.log(delta_A) - np.sum(np.log(uncerts)) +\
    (0.5 * np.log(np.pi/2)) - (chisq_W/2) + np.log(erfc_term)

    return log_P_D


def marginalize_Phi_M(log_P_D, Omega, Phi, M, omega_min, omega_max):
    #* Marginalize the given log prob over phi and M, adding in the priors on omega and phi.
    #* Leaves the omega dimension intact for use in frequency detection
    log_omega_prior = np.log(1 / (Omega * np.log(omega_max/omega_min)))
    log_phi_prior = np.log(1 / (2 * np.pi))

    log_P = log_P_D + log_omega_prior[:, np.newaxis, np.newaxis] + log_phi_prior

    #* Integrate over Phi and M to get P(D|w, b=1, I)
    log_offset_factor = np.max(log_P)
    P = np.exp(log_P - log_offset_factor)  # Subtract max for numerical stability

    P_over_M = simpson(P, x=M, axis=-1)  # Integrate over M
    P_over_Phi_M = simpson(P_over_M, x=Phi, axis=-1)  # Integrate over Phi

    return P_over_Phi_M, log_offset_factor #+ np.max(log_P_D)  # Reapply max factor
    
def log_odds_ratio(log_P_D_periodic, log_P_D_nonperiodic, Omega, Phi, M, omega_min, omega_max):

    P_over_Phi_M, log_offset_factor = marginalize_Phi_M(log_P_D_periodic, Omega, Phi, M, omega_min, omega_max)

    P_periodic = simpson(P_over_Phi_M, x=Omega)

    return np.log(P_periodic) -  (log_P_D_nonperiodic -log_offset_factor)

# ---------------------------------------------
# Outdated functions kept for reference

# def Omega_Prior(Omega):
#     w_start = 1
#     w_end = 20 #! Change if omega range changes
#     return 1/(np.log(w_end/w_start)/Omega)

# def M_prior(M):
#     return loggamma(M+1)

# def CalculateProbs(Omega, Phi, M, t, signal, time_total, dt):
#     P = np.zeros((len(Omega), len(Phi), len(M)))
#     N = np.sum(signal)

#     Pr_omega = Omega_Prior(Omega)
#     Pr_M = M_prior(M)

#     for i, w in enumerate(Omega):
#         for j, phi in enumerate(Phi):
#             for k, m in enumerate(M):
#                 fold_summed_signal, bin_phases = phase_fold(t, signal, omega=w, phi=phi, num_bins=m)
#                 n = fold_summed_signal
#                 r = n*m/time_total
#                 # n_fake = r * time_total/m
#                 A = np.mean(r)
#                 Am = A*m
#                 f = r/(Am)
#                 f = np.clip(f, 1e-12, 1)
#                 # if Am < 0:
#                 #     print(Am, "Am is negative")
#                 logP_D = N*(np.log(dt) + np.log(Am)) +\
#                         np.dot(n, np.log(f)) - A*time_total +\
#                         np.log(Pr_M[k]) + np.log(Pr_omega[i])
#                 P[i,j,k] = logP_D
                
#     return P

# def Null_Likelihood(t, signal, time_total, dt):
#     N = np.sum(signal)
#     A  = np.mean(signal)/dt #* m reduces to the number of timesteps for unbinned, so m/time_total = 1/dt
#     log_P_D = N * (np.log(dt) + np.log(A)) - A*time_total
#     return log_P_D

# def Omega_Curve(t, signal, omega_min, omega_max, omega_steps, phi_steps=20, m_min=5, m_max=20, return_likelihood_ratio=False):
#     Omega = np.linspace(omega_min, omega_max, omega_steps)
#     Phi = np.linspace(0, 2*np.pi, phi_steps)
#     M = np.arange(m_min, m_max+1)
    
#     time_total = t.max() - t.min()
#     dt = np.mean(np.diff(t)) #! Only work for uniform or uniform-ish timesteps

#     P = CalculateProbs(Omega, Phi, M, t, signal, time_total, dt)

#     Y     = np.linspace(0, 2*np.pi, len(Phi))
#     Z     = np.linspace(M.min()  , M.max(), len(M))

#     margM = simpson(P, x = Z)#marginalization over bins
#     Pw    = simpson(margM, x = Y, axis = 1)#best omega marginalization over bins phase

#     if return_likelihood_ratio:
#         log_P_D_unperiodic = Null_Likelihood(t, signal, time_total, dt)
#         log_P_D_periodic = simpson(Pw, x=Omega)
#         print(log_P_D_periodic, log_P_D_unperiodic)
#         likelihood_ratio = log_P_D_periodic - log_P_D_unperiodic

#         return Omega, Pw, likelihood_ratio

#     return Omega, Pw

