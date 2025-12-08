import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import simpson
from scipy.signal import find_peaks
from scipy.special import loggamma

def phase_fold(t, signal, omega, phi, num_bins):
    phase = ((t * omega + phi) / (2 * np.pi)) % 1
    timestamp_bins = np.floor(phase * num_bins).astype(int)

    bin_signal = np.bincount(timestamp_bins, weights=signal)

    timestamps_per_bin = np.bincount(timestamp_bins)

    # if np.any(timestamps_per_bin <= 0):
    #     raise ValueError(f"Zero timestamps_per_bin encountered: omega={omega}, phi={phi}, num_bins={num_bins}")

    # bin_average = bin_signal / timestamps_per_bin

    bin_phases = np.linspace(0, 1, num_bins)

    return bin_signal, bin_phases


def Omega_Prior(Omega):
    w_start = 1
    w_end = 20
    return 1/(np.log(w_end/w_start)/Omega)

def M_prior(M):
    return loggamma(M+1)

def CalculateProbs(Omega, Phi, M, t, signal, time_total, dt):
    P = np.zeros((len(Omega), len(Phi), len(M)))
    N = np.sum(signal)

    Pr_omega = Omega_Prior(Omega)
    Pr_M = M_prior(M)

    for i, w in enumerate(Omega):
        for j, phi in enumerate(Phi):
            for k, m in enumerate(M):
                fold_summed_signal, bin_phases = phase_fold(t, signal, omega=w, phi=phi, num_bins=m)
                n = fold_summed_signal
                r = n*m/time_total
                # n_fake = r * time_total/m
                A = np.mean(r)
                Am = A*m
                f = r/(Am)
                f = np.clip(f, 1e-12, 1)
                # if Am < 0:
                #     print(Am, "Am is negative")
                logP_D = N*(np.log(dt) + np.log(Am)) +\
                        np.dot(n, np.log(f)) - A*time_total +\
                        np.log(Pr_M[k]) + np.log(Pr_omega[i])
                P[i,j,k] = logP_D
                
    return P

def Null_Likelihood(t, signal, time_total, dt):
    N = np.sum(signal)
    A  = np.mean(signal)/dt #* m reduces to the number of timesteps for unbinned, so m/time_total = 1/dt
    log_P_D = N * (np.log(dt) + np.log(A)) - A*time_total
    return log_P_D

def Omega_Curve(t, signal, omega_min, omega_max, omega_steps, phi_steps=20, m_min=5, m_max=20, return_likelihood_ratio=False):
    Omega = np.linspace(omega_min, omega_max, omega_steps)
    Phi = np.linspace(0, 2*np.pi, phi_steps)
    M = np.arange(m_min, m_max+1)
    
    time_total = t.max() - t.min()
    dt = np.mean(np.diff(t)) #! Only work for uniform or uniform-ish timesteps

    P = CalculateProbs(Omega, Phi, M, t, signal, time_total, dt)

    Y     = np.linspace(0, 2*np.pi, len(Phi))
    Z     = np.linspace(M.min()  , M.max(), len(M))

    margM = simpson(P, x = Z)#marginalization over bins
    Pw    = simpson(margM, x = Y, axis = 1)#best omega marginalization over bins phase

    if return_likelihood_ratio:
        log_P_D_unperiodic = Null_Likelihood(t, signal, time_total, dt)
        log_P_D_periodic = simpson(Pw, x=Omega)
        print(log_P_D_periodic, log_P_D_unperiodic)
        likelihood_ratio = log_P_D_periodic - log_P_D_unperiodic

        return Omega, Pw, likelihood_ratio

    return Omega, Pw


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
    return t, np.clip(signal + np.random.normal(0, noise_ratio, n_timesteps), 0, 1)

def best_omega(Omega, Pw):
    return Omega[Pw.argmax()]