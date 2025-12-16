
#* This module contains functions to perform Bayesian periodic signal detection using the method outlined in https://iopscience.iop.org/article/10.1086/307433/pdf (Gregory 1999)

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
from numba import njit, prange

def phase_fold(t, signal, omega, phi, num_bins, reduction='sum'):
    phase = ((t * omega + phi) / (2 * np.pi)) % 1
    timestamp_bins = np.floor(phase * num_bins).astype(int)

    bin_signal = np.bincount(timestamp_bins, weights=signal, minlength=num_bins)

    timestamps_per_bin = np.bincount(timestamp_bins, minlength=num_bins)

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
    #* Make a simple synthetic periodic signal for testing (not specific to exoplanets)

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

def in_transit(times, omega, phi, f_trans):
    phase = ((times * omega + phi) / (2 * np.pi)) % 1
    return phase < f_trans

def bin_sum(values, bins, m):
    return np.bincount(bins, weights=values, minlength=m)

# Using algorithm from https://iopscience.iop.org/article/10.1086/307433/pdf (Gregory 1999)
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
    #* A parallelized version of calculate_logprob using threads
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
    print("PARALLEL EXECUTION: Using n_jobs =", n_jobs, "with chunk_size =", chunk_size)

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

@njit(parallel=True, fastmath=True, cache=True)
def _calculate_logprob_numba_kernel(t, Omega, Phi, M_int, one_over_square_uncerts, d_Wj_unbinsummed, d_Wj2_unbinsummed, r_min, r_max, sum_log_uncerts, delta_r, N):
	num_omega = Omega.shape[0]
	num_phi = Phi.shape[0]
	num_m = M_int.shape[0]
	out = np.zeros((num_omega, num_phi, num_m))
	two_pi = 2.0 * math.pi
	log_two_pi = math.log(2.0 * math.pi)
	log_pi_over_2 = math.log(math.pi / 2.0)
	for i in prange(num_omega):
		w = Omega[i]
		for j in range(num_phi):
			p = Phi[j]
			for k in range(num_m):
				m = int(M_int[k])
				# Compute timestamp bins
				bins = np.empty(t.shape[0], dtype=np.int64)
				for idx in range(t.shape[0]):
					phase = (t[idx] * w + p) % two_pi
					bins[idx] = int(math.floor(m * (phase / two_pi)))
                         
				
				counts = np.bincount(bins, minlength=m)
				w_j_mask = counts > 1
				

				W_j = np.bincount(bins, weights=one_over_square_uncerts, minlength=m)
				for b in range(m):
					if not w_j_mask[b]:
						W_j[b] = 1.0
				d_Wj_num = np.bincount(bins, weights=d_Wj_unbinsummed, minlength=m)
				d_Wj2_num = np.bincount(bins, weights=d_Wj2_unbinsummed, minlength=m)
				d_Wj = d_Wj_num / W_j
				d_Wj2 = d_Wj2_num / W_j
				sum_wj = 0
				for b in range(m):
					if w_j_mask[b]:
						sum_wj += 1
				scale = 0.0
				if sum_wj > 0:
					scale = (1.0 * m) / (1.0 * sum_wj)
				

				sum_chisq_over2 = 0.0
				sum_log_erfc = 0.0
				for b in range(m):
					chisq = W_j[b] * (d_Wj2[b] - d_Wj[b] * d_Wj[b]) * (1.0 if w_j_mask[b] else 0.0) * scale
					sum_chisq_over2 += 0.5 * chisq
					y_min = math.sqrt(0.5 * W_j[b]) * (r_min - d_Wj[b])
					y_max = math.sqrt(0.5 * W_j[b]) * (r_max - d_Wj[b])
					term = (math.erfc(y_min) - math.erfc(y_max)) / math.sqrt(W_j[b])
					sum_log_erfc += math.log(term)
				out[i, j, k] = (-0.5 * N * log_two_pi) + (-m * math.log(delta_r)) - sum_log_uncerts + (0.5 * m * log_pi_over_2) - sum_chisq_over2 + sum_log_erfc
	return out

def calculate_logprob_numba(t, signal, uncerts, Omega, Phi, M, r_min, r_max):
	"""Numba-accelerated version of calculate_logprob with identical math to the current implementation.
	"""
	# Ensure contiguous arrays and expected dtypes
	t_arr = np.ascontiguousarray(t, dtype=np.float64)
	signal_arr = np.ascontiguousarray(signal, dtype=np.float64)
	uncerts_arr = np.ascontiguousarray(uncerts, dtype=np.float64)
	Omega_arr = np.ascontiguousarray(Omega, dtype=np.float64)
	Phi_arr = np.ascontiguousarray(Phi, dtype=np.float64)
	M_int = np.ascontiguousarray(M.astype(np.int64))
	# Precomputations (same as fast path)
	one_over_square_uncerts = np.ascontiguousarray(1.0 / (uncerts_arr * uncerts_arr), dtype=np.float64)
	d_Wj_unbinsummed = np.ascontiguousarray(signal_arr / (uncerts_arr * uncerts_arr), dtype=np.float64)
	d_Wj2_unbinsummed = np.ascontiguousarray((signal_arr / uncerts_arr) * (signal_arr / uncerts_arr), dtype=np.float64)
	sum_log_uncerts = float(np.sum(np.log(uncerts_arr)))
	delta_r = float(r_max - r_min)
	N = int(t_arr.shape[0])
	return _calculate_logprob_numba_kernel(
		t_arr,
		Omega_arr,
		Phi_arr,
		M_int,
		one_over_square_uncerts,
		d_Wj_unbinsummed,
		d_Wj2_unbinsummed,
		float(r_min),
		float(r_max),
		sum_log_uncerts,
		delta_r,
		N,
	)

def calculate_logprob_transit_model(t, signal, uncerts, Omega, Phi, F_trans, r_min, r_max):
    #* Variant model using two bins (one transit bin, one non-transit bin). Promising, but ruled out by increased compute time in numerical marginalization
    delta_r = r_max - r_min
    N = len(t)
    m=2

    log_P_D = np.zeros((len(Omega), len(Phi), len(F_trans)))
    bar = progressbar.ProgressBar(maxval=len(Omega))
    for i, w in bar(enumerate(Omega)):
        for j, p in enumerate(Phi):
            for k, f_trans in enumerate(F_trans):

                in_trans = in_transit(t, w, p, f_trans)

                w_j = np.bincount(in_trans, minlength=m) > 1 #* see appendix of paper, deals with when bins have < 2 samples
                
                W_j = bin_sum(1/np.square(uncerts), in_trans, m)

                W_j[w_j==0] = 1.0 #* Modification from appendix, set to 1

                # if np.any(W_j == 0):
                #     raise ValueError(f"Zero W_j encountered: omega={w}, phi={p}, m={m}")
                
                d_Wj = bin_sum(signal/np.square(uncerts), in_trans, m)/W_j

                d_Wj2 = bin_sum(np.square(signal/uncerts), in_trans, m)/W_j

                chisq_Wj = W_j * (d_Wj2 - np.square(d_Wj)) * w_j / (1/m * np.sum(w_j)) #* modification to chisq_Wj from (A1) in appendix

                y_jmin = np.sqrt(0.5 * W_j) * (r_min - d_Wj)
                y_jmax = np.sqrt(0.5 * W_j) * (r_max - d_Wj)

                erfc_terms = 1/np.sqrt(W_j) * (erfc(y_jmin) - erfc(y_jmax))

                log_P_D[i, j, k] = (-0.5*N*np.log(2*np.pi)) + (-m * np.log(delta_r)) - np.sum(np.log(uncerts)) +\
                (0.5 * m * np.log(np.pi/2)) - np.sum(chisq_Wj/2) + np.sum(np.log(erfc_terms))
                
    return log_P_D

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


def P_nonperiodic(t, signal, uncerts, Phi, M, r_min, r_max):
    #*  Variant of eqn 32, 33 from paper, (for non-periodic model, meaning time window just contains one period)
    omega = np.array([2 * np.pi / (t.max() - t.min())])
    log_P_D = calculate_logprob(t, signal, uncerts, omega, Phi, M, r_min, r_max)

    #* Marginalize over phi and M without adding Omega prior (since omega is fixed here)
    log_phi_prior = np.log(1 / (2 * np.pi))
    log_P = log_P_D + log_phi_prior

    log_offset_factor = np.max(log_P)
    P = np.exp(log_P - log_offset_factor)  # Subtract max for numerical stability

    P_over_M = simpson(P, x=M, axis=-1)  # Integrate over M
    P_over_Phi_M = simpson(P_over_M, x=Phi, axis=-1)  # Integrate over Phi

    if len(P_over_Phi_M.flatten()) != 1:
        raise ValueError("Expected single value for non-periodic log prob after marginalization")
    
    return P_over_Phi_M[0], log_offset_factor


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

    return P_over_Phi_M, log_offset_factor

def marginalize_Phi_F_trans(log_P_D, Omega, Phi, F_trans, omega_min, omega_max):
    #* Marginalization function for the transit-specific periodic model
    log_omega_prior = np.log(1 / (Omega * np.log(omega_max/omega_min)))
    log_phi_prior = np.log(1 / (2 * np.pi))
    log_F_trans_prior = np.zeros_like(F_trans)

    log_P = log_P_D + log_omega_prior[:, np.newaxis, np.newaxis] + log_phi_prior[np.newaxis, :, np.newaxis] + log_F_trans_prior[np.newaxis, np.newaxis, :]

    #* Integrate over Phi and M to get P(D|w, b=1, I)
    log_offset_factor = np.max(log_P)
    P = np.exp(log_P - log_offset_factor)  # Subtract max for numerical stability

    P_over_F = simpson(P, x=F_trans, axis=-1)  # Integrate over M
    P_over_Phi_F = simpson(P_over_F, x=Phi, axis=-1)  # Integrate over Phi

    return P_over_Phi_F, log_offset_factor
    
def log_odds_ratio_constant(log_P_D_periodic, log_P_D_nonperiodic, Omega, Phi, M, omega_min, omega_max):
    #* Get the odds ratio between the periodic model and the constant model. No prior preference between the models is assumed.

    P_over_Phi_M, log_offset_factor = marginalize_Phi_M(log_P_D_periodic, Omega, Phi, M, omega_min, omega_max)

    P_periodic = simpson(P_over_Phi_M, x=Omega)

    return np.log(P_periodic) -  (log_P_D_nonperiodic -log_offset_factor)

def log_odds_ratio_nonperiodic(log_P_D_periodic, P_nonperiodic, log_offset_nonperiodic, Omega, Phi, M, omega_min, omega_max):
    #* Get the odds ratio between the periodic model and the non-periodic model. No prior preference between the models is assumed.

    P_over_Phi_M, log_offset_periodic = marginalize_Phi_M(log_P_D_periodic, Omega, Phi, M, omega_min, omega_max)

    P_periodic = simpson(P_over_Phi_M, x=Omega)

    return (np.log(P_periodic) + log_offset_periodic) -  (np.log(P_nonperiodic) + log_offset_nonperiodic)

def optimal_spaced_omega(omega_min, omega_max, total_time, transit_duration):
    """Calculate optimal spacing of omega values to ensure no peak in P(w|D) is missed. See report for derivation."""
    N = int(np.ceil(np.log(omega_max/omega_min) / np.log(1+ transit_duration/total_time)))
    n = np.arange(0, N+1)
    return omega_min * np.pow(1 + transit_duration/total_time, n)