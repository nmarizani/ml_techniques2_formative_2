#!/usr/bin/env python3
"""
combine_csv.py

This version works with a simpler folder layout like:
Data/
├── Jumping/
│   ├── jumping_1.csv
│   ├── jumping_2.csv
├── Standing/
├── Still/
├── Walking/

It merges all CSVs per activity into a single combined file in "Data/combined".
"""

import os
import re
import sys
import pandas as pd
import argparse

ACCEL_KEYWORDS = ("accelerometer", "accel")
GYRO_KEYWORDS = ("gyroscope", "gyro")
OUTPUT_COLS = ["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


def find_activities(source_root: str):
    """Return all activity directories under the source root."""
    return [os.path.join(source_root, d) for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]


def find_sensor_files_in_activity(activity_dir: str):
    """Return lists of accelerometer and gyroscope CSVs in the given activity folder."""
    accel_files, gyro_files = [], []
    for root, _, files in os.walk(activity_dir):
        for f in files:
            low = f.lower()
            path = os.path.join(root, f)
            if low.endswith(".csv") and any(k in low for k in ACCEL_KEYWORDS):
                accel_files.append(path)
            elif low.endswith(".csv") and any(k in low for k in GYRO_KEYWORDS):
                gyro_files.append(path)
    return accel_files, gyro_files


def choose_timestamp_column(df: pd.DataFrame):
    for pattern in (r"time", r"timestamp", r"ts", r"seconds", r"sec", r"elapsed"):
        for c in df.columns:
            if re.search(pattern, c, flags=re.I):
                return c
    return df.columns[0]


def detect_xyz_columns(df: pd.DataFrame):
    mapping = {}
    for col in df.columns:
        low = col.lower()
        if "time" in low or "timestamp" in low:
            continue
        for axis in ("x", "y", "z"):
            if re.search(rf"[^a-z]?{axis}[^a-z]?", low):
                mapping[axis] = col
    if len(mapping) < 3:
        raise ValueError(f"Could not detect x,y,z columns in {list(df.columns)}")
    return mapping


def merge_session(accel_path: str, gyro_path: str):
    a = pd.read_csv(accel_path)
    g = pd.read_csv(gyro_path)

    ts_a = choose_timestamp_column(a)
    ts_g = choose_timestamp_column(g)

    common_ts = "timestamp"
    a = a.rename(columns={ts_a: common_ts})
    g = g.rename(columns={ts_g: common_ts})

    am = detect_xyz_columns(a)
    gm = detect_xyz_columns(g)

    a_sel = a[[common_ts, am["x"], am["y"], am["z"]]].rename(columns={am["x"]: "accel_x", am["y"]: "accel_y", am["z"]: "accel_z"})
    g_sel = g[[common_ts, gm["x"], gm["y"], gm["z"]]].rename(columns={gm["x"]: "gyro_x", gm["y"]: "gyro_y", gm["z"]: "gyro_z"})

    merged = pd.merge(a_sel, g_sel, on=common_ts, how="inner")
    return merged


def sanitize_name(name: str):
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip().lower())
    return re.sub(r"_+", "_", s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", "-s", default="data", help="Folder containing activity subfolders")
    parser.add_argument("--out", "-o", default=os.path.join("data", "combined"), help="Output folder")
    parser.add_argument("--name", "-n", help="Optional name prefix for output files")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    name_token = sanitize_name(args.name or "")

    activities = find_activities(args.source_root)
    if not activities:
        print(f"No activity folders found under {args.source_root}")
        sys.exit(1)

    for activity_dir in activities:
        activity_name = os.path.basename(activity_dir)
        accel_files, gyro_files = find_sensor_files_in_activity(activity_dir)

        if not accel_files or not gyro_files:
            print(f"⚠️  Skipping {activity_name}: missing accel or gyro CSVs")
            continue

        merged_rows = []
        for a_file, g_file in zip(accel_files, gyro_files):
            try:
                merged = merge_session(a_file, g_file)
                merged_rows.append(merged)
            except Exception as e:
                print(f"Error merging {a_file} & {g_file}: {e}")

        if not merged_rows:
            print(f"⚠️  No valid merged data for {activity_name}")
            continue

        df = pd.concat(merged_rows, ignore_index=True)
        if "timestamp" in df.columns:
            df = df.sort_values(by="timestamp")

        out_name = f"{name_token}_{activity_name}_combined.csv" if name_token else f"{activity_name}_combined.csv"
        out_path = os.path.join(args.out, out_name)
        df.to_csv(out_path, index=False, columns=[c for c in OUTPUT_COLS if c in df.columns])
        print(f"✅ Wrote: {out_path} ({df.shape[0]} rows)")

    print("🎉 Done. All combined files saved under:", args.out)


if __name__ == "__main__":
    main()