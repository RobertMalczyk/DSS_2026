"""Blind denoising pipeline for three unknown signals.

Entry point: denoise(noisy, fs) -> filtered array.
Dispatches on fs to a per-signal chain. All cutoffs / Qs / orders inside each
chain were derived from the inspection pass in inspect_signals.py /
inspect_detail.py — see summary.md for the measurements that motivated them.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sps

np.random.seed(42)


# ---------------------------------------------------------------------------
# filter factories (all zero-phase via sosfiltfilt at call site)
# ---------------------------------------------------------------------------

def _butter_sos(order, cutoff, btype, fs):
    return sps.butter(order, cutoff, btype=btype, fs=fs, output="sos")


def _notch_sos(f0, Q, fs):
    b, a = sps.iirnotch(f0, Q, fs=fs)
    return sps.tf2sos(b, a)


def _apply(stages, x):
    y = x
    for _, sos in stages:
        y = sps.sosfiltfilt(sos, y)
    return y


# ---------------------------------------------------------------------------
# per-signal chain definitions — labelled list of (name, sos)
# ---------------------------------------------------------------------------

def _chain_ecg(fs):
    return [
        ("HP Butter ord4 @ 0.5 Hz",   _butter_sos(4, 0.5,  "high", fs)),
        ("LP Butter ord6 @ 30 Hz",    _butter_sos(6, 30.0, "low",  fs)),
    ]


def _chain_vibration(fs):
    return [
        ("HP Butter ord4 @ 5 Hz",     _butter_sos(4, 5.0, "high", fs)),
        ("Notch 50 Hz  Q=30",         _notch_sos(50.0,  30, fs)),
        ("Notch 100 Hz Q=30",         _notch_sos(100.0, 30, fs)),
        ("Notch 150 Hz Q=30",         _notch_sos(150.0, 30, fs)),
        ("LP Butter ord8 @ 2800 Hz",  _butter_sos(8, 2800.0, "low", fs)),
    ]


def _chain_music(fs):
    return [
        ("HP Butter ord4 @ 25 Hz",    _butter_sos(4, 25.0, "high", fs)),
        ("Notch 50 Hz  Q=30",         _notch_sos(50.0,   30, fs)),
        ("Notch 100 Hz Q=30",         _notch_sos(100.0,  30, fs)),
        ("Notch 4410 Hz Q=110",       _notch_sos(4410.0, 110, fs)),
    ]


def get_chain(fs: float) -> list[tuple[str, np.ndarray]]:
    """Return the labelled SOS stages for the chain selected by fs.

    Exposed so make_bode.py and other diagnostics can plot the same stages
    denoise() applies."""
    if fs <= 500:
        return _chain_ecg(fs)
    if fs <= 16000:
        return _chain_vibration(fs)
    return _chain_music(fs)


# ---------------------------------------------------------------------------
# single entry point — required signature (noisy, fs)
# ---------------------------------------------------------------------------

def denoise(noisy: np.ndarray, fs: float) -> np.ndarray:
    """Blind denoise. Selects chain by sample rate (an input), then applies
    zero-phase Butterworth / IIR notch stages whose parameters were measured
    from the inspection of the three noisy signals."""
    x = np.asarray(noisy, dtype=np.float64).ravel()
    return _apply(get_chain(fs), x)
