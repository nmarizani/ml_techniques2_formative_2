import os
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import welch

# CONFIGURATION
WINDOW_SIZE = 128
OVERLAP = 0.5
INPUT_DIR = "data/combined"
OUTPUT_DIR = "data/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def signal_magnitude_area(df, cols):
    """Compute Signal Magnitude Area (SMA)."""
    return np.mean(np.abs(df[cols]).sum(axis=1))


def dominant_frequency(signal, fs=50):
    """Return dominant frequency in Hz using FFT."""
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    fft_vals = np.abs(fft(signal))
    return freqs[np.argmax(fft_vals)]


def spectral_energy(signal):
    """Compute total spectral energy using Welch’s method."""
    f, Pxx = welch(signal, fs=50)
    return np.sum(Pxx)


def extract_features_from_window(window):
    """Compute both time and frequency domain features for a given window."""
    features = {}

    for prefix in ["accel", "gyro"]:
        cols = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
        if not all(c in window.columns for c in cols):
            continue

        # Time-domain
        for axis in cols:
            features[f"{axis}_mean"] = window[axis].mean()
            features[f"{axis}_std"] = window[axis].std()
            features[f"{axis}_var"] = window[axis].var()
            features[f"{axis}_mad"] = (window[axis] - window[axis].mean()).abs().mean()

        # Correlation between axes
        features[f"{prefix}_xy_corr"] = window[cols[0]].corr(window[cols[1]])
        features[f"{prefix}_xz_corr"] = window[cols[0]].corr(window[cols[2]])
        features[f"{prefix}_yz_corr"] = window[cols[1]].corr(window[cols[2]])

        # Signal Magnitude Area
        features[f"{prefix}_sma"] = signal_magnitude_area(window, cols)

        # Frequency-domain
        for axis in cols:
            sig = window[axis].values
            features[f"{axis}_dom_freq"] = dominant_frequency(sig)
            features[f"{axis}_spec_energy"] = spectral_energy(sig)

    return features


def sliding_windows(data, window_size, overlap):
    """Yield start and end indices for overlapping windows."""
    step = int(window_size * (1 - overlap))
    for start in range(0, len(data) - window_size + 1, step):
        yield start, start + window_size


def process_activity_file(filepath, label):
    """Read a combined CSV and extract windowed features."""
    df = pd.read_csv(filepath)
    feature_rows = []

    for start, end in sliding_windows(df, WINDOW_SIZE, OVERLAP):
        window = df.iloc[start:end]
        feats = extract_features_from_window(window)
        feats["activity"] = label
        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)


def main():
    print("Extracting features from combined files...")
    all_features = []

    for file in os.listdir(INPUT_DIR):
        if file.endswith(".csv"):
            label = os.path.basename(file).split("_")[1]  # e.g., nicolle_jumping_combined.csv → 'jumping'
            filepath = os.path.join(INPUT_DIR, file)
            print(f"→ Processing {file} ({label})")
            df_features = process_activity_file(filepath, label)
            all_features.append(df_features)

    final_df = pd.concat(all_features, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "features.csv")
    final_df.to_csv(out_path, index=False)
    print(f"Features saved to: {out_path}")
    print(f"Total windows: {len(final_df)}")


if __name__ == "__main__":
    main()