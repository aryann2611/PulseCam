"""Smallest checks that fail if the HR maths breaks. Run: python -m pytest tests"""
import numpy as np

from pulsecam.metrics import mae, rmse, summary
from pulsecam.signal import fft_bpm, pos_bvp

FS, BPM = 30, 72
t = np.arange(10 * FS) / FS
rng = np.random.default_rng(0)


def test_fft_bpm_finds_pulse_in_noise():
    bpm, snr = fft_bpm(np.sin(2 * np.pi * BPM / 60 * t) + 0.5 * rng.standard_normal(len(t)), FS)
    assert abs(bpm - BPM) < 1.5 and snr > 0


def test_fft_bpm_ignores_out_of_band_drift():
    bpm, _ = fft_bpm(5 * np.sin(2 * np.pi * 0.2 * t) + np.sin(2 * np.pi * BPM / 60 * t), FS)  # 12 BPM drift
    assert abs(bpm - BPM) < 1.5


def test_pos_recovers_pulse_from_rgb():
    pulse = np.sin(2 * np.pi * BPM / 60 * t)
    skin = np.array([180., 130., 110.])
    rgb = skin * (1 + 0.002 * pulse[:, None] * np.array([0.3, 1.0, 0.5])) + rng.standard_normal((len(t), 3)) * 0.05
    assert abs(fft_bpm(pos_bvp(rgb, FS), FS)[0] - BPM) < 2


def test_metrics():
    assert mae([70, 80], [72, 76]) == 3 and abs(rmse([70, 80], [72, 76]) - np.sqrt(10)) < 1e-9
    assert summary([60, 70, 80], [60, 70, 80])["Pearson"] == 1.0
