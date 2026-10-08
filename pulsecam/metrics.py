"""Evaluation metrics (same definitions as rPPG-Toolbox) and the Bland-Altman plot."""
import numpy as np


def mae(p, t):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(t))))


def rmse(p, t):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(t)) ** 2)))


def mape(p, t):
    p, t = np.asarray(p, float), np.asarray(t, float)
    return float(np.mean(np.abs(p - t) / t) * 100)


def pearson(p, t):
    return float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 else float("nan")


def summary(p, t):
    return dict(n=len(p), MAE=mae(p, t), RMSE=rmse(p, t), MAPE=mape(p, t), Pearson=pearson(p, t))


def bland_altman(ax, p, t, title=""):
    p, t = np.asarray(p, float), np.asarray(t, float)
    m, d = (p + t) / 2, p - t
    bias, sd = d.mean(), (d.std(ddof=1) if len(d) > 1 else 0.0)
    ax.scatter(m, d, s=18)
    for y, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
        ax.axhline(y, color="k", ls=ls, lw=1)
    ax.set(xlabel="mean of PulseCam and reference (BPM)", ylabel="PulseCam - reference (BPM)",
           title=f"{title}  bias {bias:.1f}, ±1.96 SD {1.96 * sd:.1f}")
