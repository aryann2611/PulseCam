"""Bandpass + FFT heart rate, SNR, and the classical POS method (Wang et al., 2016)."""
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve

LO, HI = 0.7, 3.0  # Hz, i.e. 42-180 BPM


def bandpass(x, fs, lo=LO, hi=HI, order=3):
    b, a = butter(order, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, detrend(np.asarray(x, float)))


def smooth_detrend(x, lam=100):
    """Remove slow drift with the smoothness-priors detrend the rPPG-Toolbox uses (Tarvainen et al., 2002).

    TS-CAN outputs the derivative of the pulse; we integrate it with cumsum, which turns noise into strong slow drift
    that otherwise leaks into the low end of the heart-rate band. Same result as the toolbox's `_detrend`, but sparse."""
    x = np.asarray(x, float)
    n = len(x)
    D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    return x - spsolve((identity(n) + lam ** 2 * (D.T @ D)).tocsc(), x)


def fft_bpm(x, fs):
    """Return (bpm, snr_db). HR = 60 * peak frequency inside 0.7-3.0 Hz."""
    x = bandpass(x, fs)
    n = 1 << int(np.ceil(np.log2(len(x) * 8)))  # zero-pad for ~0.01 Hz resolution
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    band = (f >= LO) & (f <= HI)
    fp = f[band][p[band].argmax()]
    near = (np.abs(f - fp) < 0.1) | (np.abs(f - 2 * fp) < 0.1)  # peak + 2nd harmonic
    snr = 10 * np.log10(p[band & near].sum() / max(p[band & ~near].sum(), 1e-12))
    return 60 * fp, snr


def pos_bvp(rgb, fs, win_s=1.6):
    """POS pulse signal from per-frame mean RGB, shape (T, 3) -> (T,)."""
    rgb = np.asarray(rgb, float)
    T, L = len(rgb), int(win_s * fs)
    h = np.zeros(T)
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    for n in range(L, T + 1):
        w = rgb[n - L:n]
        c = w.T / np.maximum(w.mean(0), 1e-6)[:, None]
        s = P @ c
        v = s[0] + s[0].std() / max(s[1].std(), 1e-6) * s[1]
        h[n - L:n] += v - v.mean()
    return h
