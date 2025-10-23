#!/usr/bin/env python3
"""
Feature extraction (documented in the same clear style you like)

- Reads CSV files from data/combined/
- Windowing uses WINDOW_SIZE and OVERLAP (same logic as your original)
- Computes time-domain and frequency-domain features
- Adds the missing features required by the assignment:
    * resultant acceleration mean/std
    * resultant dominant frequency and spectral energy
    * top-3 FFT magnitudes and frequencies (per-axis and resultant)
- Saves a single combined output CSV: data/features/features.csv
  (per-file feature CSVs have been removed as requested)
"""

# ===================================================
# Imports
# ===================================================
import os
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch

# ===================================================
# Configuration (easy to change)
# ===================================================
WINDOW_SIZE = 128            # samples per window (you used this)
OVERLAP = 0.5                # fraction overlap (0..1)
INPUT_DIR = "data/combined"  # raw CSVs
OUTPUT_DIR = "data/features" # where to write features
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 50.0                    # sampling frequency (Hz) used for FFT/Welch
TOP_K = 3                    # how many top FFT components to save

# ===================================================
# Small helper functions (simple and readable)
# ===================================================
def signal_magnitude_area(df, cols):
    """Compute Signal Magnitude Area (SMA) for given columns."""
    return np.mean(np.abs(df[cols]).sum(axis=1))

def _maybe_convert_timestamp(ts):
    """
    Very small heuristic to convert large integer timestamps to seconds.
    If timestamps look like nanoseconds (very large), convert to seconds.
    Keep it simple: if mean > 1e12 treat as ns and divide by 1e9.
    """
    if ts is None:
        return None
    try:
        arr = np.array(ts, dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None
    meanv = float(np.nanmean(arr))
    if meanv > 1e12:
        return arr / 1e9
    if meanv > 1e9:
        return arr / 1e3
    return arr

def dominant_frequency(signal, fs=FS):
    """Return dominant frequency in Hz using FFT (exclude DC)."""
    if len(signal) < 2:
        return 0.0
    yf = np.abs(rfft(signal))
    xf = rfftfreq(len(signal), 1.0 / fs)
    if yf.size <= 1:
        return 0.0
    idx = int(np.argmax(yf[1:]) + 1)
    return float(xf[idx])

def spectral_energy(signal, fs=FS):
    """Compute spectral energy using Welch's PSD."""
    if len(signal) < 2:
        return 0.0
    f, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    return float(np.sum(Pxx))

def fft_top_k(signal, k=TOP_K, fs=FS):
    """
    Return top-k FFT magnitudes and their frequencies (excluding DC).
    Returns two lists (mags, freqs), each length k (padded with zeros if needed).
    """
    if len(signal) < 2:
        return [0.0]*k, [0.0]*k
    yf = np.abs(rfft(signal))
    xf = rfftfreq(len(signal), 1.0 / fs)
    if yf.size <= 1:
        return [0.0]*k, [0.0]*k
    mags = yf[1:]
    freqs = xf[1:]
    order = np.argsort(mags)[::-1]
    top_mags = []
    top_freqs = []
    for i in range(k):
        if i < order.size:
            top_mags.append(float(mags[order[i]]))
            top_freqs.append(float(freqs[order[i]]))
        else:
            top_mags.append(0.0)
            top_freqs.append(0.0)
    return top_mags, top_freqs

# ===================================================
# Core feature extraction for one window (simple)
# ===================================================
def extract_features_from_window(window):
    """
    window: pandas DataFrame of rows in the window
    returns: dict of features
    """
    features = {}

    # accel and gyro per-axis time-domain and simple freq-domain features
    for prefix in ["accel", "gyro"]:
        cols = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
        if not all(c in window.columns for c in cols):
            # skip if columns missing
            continue

        # time-domain: mean, std, var, mad
        for axis in cols:
            arr = window[axis].values
            features[f"{axis}_mean"] = float(np.mean(arr))
            features[f"{axis}_std"] = float(np.std(arr, ddof=0))
            features[f"{axis}_var"] = float(np.var(arr, ddof=0))
            features[f"{axis}_mad"] = float(np.mean(np.abs(arr - np.mean(arr))))

        # correlations
        def safe_corr(a, b):
            if np.std(a) == 0 or np.std(b) == 0:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        features[f"{prefix}_xy_corr"] = safe_corr(window[cols[0]].values, window[cols[1]].values)
        features[f"{prefix}_xz_corr"] = safe_corr(window[cols[0]].values, window[cols[2]].values)
        features[f"{prefix}_yz_corr"] = safe_corr(window[cols[1]].values, window[cols[2]].values)

        # SMA
        features[f"{prefix}_sma"] = float(signal_magnitude_area(window, cols))

        # frequency-domain per axis: dominant freq, spectral energy, top-k
        for axis in cols:
            sig = window[axis].values
            features[f"{axis}_dom_freq"] = dominant_frequency(sig, fs=FS)
            features[f"{axis}_spec_energy"] = spectral_energy(sig, fs=FS)
            top_mags, top_freqs = fft_top_k(sig, k=TOP_K, fs=FS)
            for i in range(TOP_K):
                features[f"{axis}_fft_top{i+1}_mag"] = top_mags[i]
                features[f"{axis}_fft_top{i+1}_freq"] = top_freqs[i]

    # resultant acceleration (orientation-invariant) features
    if all(c in window.columns for c in ["accel_x", "accel_y", "accel_z"]):
        ax = window["accel_x"].values
        ay = window["accel_y"].values
        az = window["accel_z"].values
        res = np.sqrt(ax**2 + ay**2 + az**2)
        features["acc_res_mean"] = float(np.mean(res))
        features["acc_res_std"] = float(np.std(res, ddof=0))
        features["acc_res_var"] = float(np.var(res, ddof=0))
        features["acc_res_sma"] = float(np.mean(np.abs(res)))
        # freq on resultant
        features["acc_res_dom_freq"] = dominant_frequency(res, fs=FS)
        features["acc_res_spec_energy"] = spectral_energy(res, fs=FS)
        top_mags_res, top_freqs_res = fft_top_k(res, k=TOP_K, fs=FS)
        for i in range(TOP_K):
            features[f"acc_res_fft_top{i+1}_mag"] = top_mags_res[i]
            features[f"acc_res_fft_top{i+1}_freq"] = top_freqs_res[i]
    else:
        # fill zeros if accel not present
        features["acc_res_mean"] = 0.0
        features["acc_res_std"] = 0.0
        features["acc_res_var"] = 0.0
        features["acc_res_sma"] = 0.0
        features["acc_res_dom_freq"] = 0.0
        features["acc_res_spec_energy"] = 0.0
        for i in range(TOP_K):
            features[f"acc_res_fft_top{i+1}_mag"] = 0.0
            features[f"acc_res_fft_top{i+1}_freq"] = 0.0

    return features

# ===================================================
# Windowing helpers (kept the same style)
# ===================================================
def sliding_windows(data_len, window_size, overlap):
    """Yield start and end indices for overlapping windows."""
    step = int(window_size * (1 - overlap))
    if step < 1:
        step = 1
    for start in range(0, data_len - window_size + 1, step):
        yield start, start + window_size

# ===================================================
# Process one CSV file (keeps start_time if present)
# ===================================================
def process_activity_file(filepath, label):
    """Read a combined CSV and extract windowed features."""
    df = pd.read_csv(filepath)
    feature_rows = []

    # convert timestamp column if it exists (simple heuristic)
    if "timestamp" in df.columns:
        ts = _maybe_convert_timestamp(df["timestamp"].values)
        if ts is not None and len(ts) == len(df):
            df["timestamp"] = ts

    has_time = "timestamp" in df.columns

    for start, end in sliding_windows(len(df), WINDOW_SIZE, OVERLAP):
        window = df.iloc[start:end].reset_index(drop=True)
        feats = extract_features_from_window(window)
        feats["activity"] = label
        # store start_time (original if present, otherwise window index)
        feats["start_time"] = float(window["timestamp"].iloc[0]) if has_time else float(start)
        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)

# ===================================================
# Main: iterate files, save combined CSV only (no per-file outputs)
# ===================================================
def main():
    print("Extracting features from combined files...")
    all_features = []

    for file in sorted(os.listdir(INPUT_DIR)):
        if not file.endswith(".csv"):
            continue
        # infer label same as before: second token after underscore
        label = os.path.basename(file).split("_")[1] if "_" in file else "unknown"
        filepath = os.path.join(INPUT_DIR, file)
        print(f"→ Processing {file} ({label})")
        df_features = process_activity_file(filepath, label)
        print(f"  Extracted {len(df_features)} windows from {file}")
        all_features.append(df_features)

    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        out_path = os.path.join(OUTPUT_DIR, "features.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Combined features saved to: {out_path}")
        print(f"Total windows: {len(final_df)}")
    else:
        print("No features extracted. Check input files and column names.")

if __name__ == "__main__":
    main()