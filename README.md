
# Human Activity Recognition using Hidden Markov Models

## Overview
This project develops a comprehensive pipeline for Human Activity Recognition (HAR) using smartphone accelerometer and gyroscope data. It employs Hidden Markov Models (HMMs) to classify activities such as standing, walking, jumping, and still by modeling temporal sequences of extracted features.

The motivation stems from the need for accurate, real-time activity recognition in applications like fitness tracking, fall detection, and healthcare monitoring. By leveraging statistical models, this project demonstrates how HMMs can infer hidden activity states from observable sensor data, capturing both the sequential nature of human movements and the uncertainties in measurements.

Key components include data preprocessing, feature engineering, HMM training, and evaluation with visualizations.

## Features
- **Data Preprocessing**: Merges raw accelerometer and gyroscope CSVs per activity, aligns timestamps, and handles data cleaning.
- **Feature Extraction**: Computes a wide range of time-domain (e.g., mean, variance, SMA) and frequency-domain (e.g., dominant frequency, spectral energy) features from overlapping windows.
- **HMM Modeling**: Implements both library-based (hmmlearn) and from-scratch supervised HMMs, including Viterbi decoding for state sequence prediction.
- **Evaluation Metrics**: Provides accuracy, sensitivity, specificity, confusion matrices, and transition heatmaps.
- **Visualizations**: Generates plots for transition matrices, confusion matrices, and predicted vs. actual timelines.

## Installation and Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nmarizani/ml_techniques2_formative_2.git
   cd ml_techniques2_formative_2
   ```

2. **Create a Virtual Environment** (recommended to avoid dependency conflicts):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not present, manually install:
   ```bash
   pip install numpy pandas scipy scikit-learn matplotlib seaborn hmmlearn joblib
   ```

4. **Verify Installation**:
   ```bash
   python -c "import numpy, pandas, hmmlearn; print('Dependencies installed successfully')"
   ```

## Data Preparation
### Raw Data Structure
Organize raw sensor data in the `data/` directory as follows:
```
data/
├── <contributor_name>/
│   ├── <activity_name>/
│   │   ├── <session_id>/
│   │   │   ├── Accelerometer.csv
│   │   │   └── Gyroscope.csv
```
- **Example**:
  ```
  data/Nicolle/Jumping/jumping_1/Accelerometer.csv
  ```
- **File Format**: CSVs with columns for timestamp, x, y, z axes.
- **Naming Convention**: Use descriptive names for activities (e.g., Jumping, Standing, Walking, Still).

### Data Collection Details
- **App Used**: Sensor Logger (Android/iOS app for recording sensor data).
- **Sampling Rate**: 70 Hz.
- **Sensors**: 
  - Accelerometer (ax, ay, az) — measures linear acceleration.
  - Gyroscope (gx, gy, gz) — measures angular velocity.
- **Activities**:
  - **Standing**: Device held steady at waist level.
  - **Walking**: Consistent pace, natural gait.
  - **Jumping**: Repeated continuous jumps.
  - **Still**: Phone placed flat on a surface (minimal motion).
- **Session Duration**: 5–10 seconds per recording, with at least 1 minute 30 seconds total per activity.
- **Total Files**: Multiple sessions per activity, labeled with contributor names for traceability.

## Usage
Run the scripts in sequence to reproduce the pipeline:

1. **Combine Raw CSVs**:
   ```bash
   python combine_csv.py -s data -o data/combined
   ```
   - Merges accel/gyro pairs by timestamp into `data/combined/<activity>_combined.csv`.
   - Handles missing data and timestamp alignment.

2. **Extract Features**:
   ```bash
   python extracted_features.py
   ```
   - Processes combined CSVs, applies windowing (128 samples, 50% overlap), and extracts features.
   - Outputs: `features.csv` with session_id, start_time, activity, and feature columns.
   - **Note**: Ensure `session_id` is present; if not, run helper scripts.

3. **Train and Evaluate HMM**:
   - Use `hidden_markov_model.ipynb` for interactive training and evaluation.
   - Outputs: Model, metrics, plots (`transition_heatmap.png`, `confusion_matrix.png`, `timeline_test_session.png`).

### Expected Outputs
- **Console**: Prints accuracy, sensitivity, specificity, and confusion matrix.
- **Files**:
  - `features.csv`: Extracted features.
  - Visualizations: Heatmaps and timelines for analysis.
  - `combined_activity_visualization_same_scale.png`: Data visualization.

## Methodology
### Preprocessing
- **Data Combination**: Uses pandas to merge accelerometer and gyroscope CSVs on timestamps, handling inner joins to ensure synchronized data.
- **Windowing**: Splits data into 128-sample windows (~1.8 seconds at 70 Hz) with 50% overlap to capture transitions while maintaining temporal resolution.

### Feature Extraction
- **Time-Domain Features** (per axis and resultant):
  - Mean: Average signal magnitude.
  - Variance/Std Dev: Measures fluctuation (distinguishes static from dynamic).
  - MAD (Mean Absolute Deviation): Robust dispersion measure.
  - SMA (Signal Magnitude Area): Total movement intensity.
  - Correlations: Between axes (e.g., xy_corr).
- **Frequency-Domain Features** (per axis and resultant):
  - Dominant Frequency: Peak in FFT spectrum (Hz).
  - Spectral Energy: Sum of squared FFT magnitudes.
  - Top-K FFT Magnitudes and Frequencies: Largest spectral components.
- **Resultant Acceleration**: Orientation-invariant features from sqrt(ax² + ay² + az²).
- **Normalization**: Z-score (StandardScaler) fitted on training data to standardize scales and stabilize covariance estimation.

### HMM Implementation
- **Model Components**:
  - Hidden States (Z): {standing, walking, jumping, still}.
  - Observations (X): Normalized feature vectors.
  - Transition Probabilities (A): Learned from sequence transitions.
  - Emission Probabilities (B): Multivariate Gaussian per state.
  - Initial Probabilities (π): Likelihood of starting states.
- **Training**: Uses `hmmlearn.GaussianHMM` or custom supervised HMM functions (`fit_supervised_hmm`, `viterbi`).
- **Decoding**: Viterbi algorithm for maximum-likelihood state sequences.
- **State Mapping**: Maps HMM states to labels via majority vote on training predictions.

## Results and Evaluation
The model was evaluated on unseen test sessions, demonstrating effective activity recognition:

- **Overall Performance**:
  - Achieved high accuracy on test data, with strong sequence alignment in timelines.
  - Sensitivity ≥93% for walking and still activities, reflecting reliable detection of rhythmic and stationary patterns.

- **Confusion Analysis**:
  - Standing often confused with still due to similar low-variance sensor signals.
  - Misclassifications primarily at activity boundaries (e.g., jumping to still) due to transient motion.

- **Transition Matrix**:
  - Realistic transitions with high self-loop probabilities (activity persistence) and plausible inter-state changes (e.g., walking → still).

- **Visual Insights**:
  - Heatmaps show learned temporal dependencies.
  - Timeline plots confirm alignment of predicted sequences with ground truth, with minor mismatches at transitions.

For full metrics, refer to generated plots and console output. Results validate HMMs for sequential HAR tasks.

## Project Structure
```
├── combine_csv.py
├── combined_activity_visualization_same_scale.png
├── data
│   ├── combined
│   │   ├── nicolle_jumping_combined.csv
│   │   ├── nicolle_standing_combined.csv
│   │   ├── nicolle_still_combined.csv
│   │   ├── nicolle_walking_combined.csv
│   │   ├── omar_jumping_combined.csv
│   │   ├── omar_standing_combined.csv
│   │   ├── omar_still_combined.csv
│   │   └── omar_walking_combined.csv
│   ├── Nicolle
│   │   ├── Jumping
│   │   │   ├── jumping_1
│   │   │   ├── jumping_10
│   │   │   └── ... (more sessions)
│   │   ├── Standing
│   │   │   ├── standing_1
│   │   │   └── ... (more sessions)
│   │   ├── Still
│   │   │   ├── still_1
│   │   │   └── ... (more sessions)
│   │   └── Walking
│   │       ├── walking_1
│   │       └── ... (more sessions)
│   └── omar
│       ├── jumping
│       │   ├── extracted
│       │   └── raw
│       ├── standing
│       │   ├── extracted
│       │   └── raw
│       ├── still
│       │   ├── extracted
│       │   └── raw
│       └── walking
│           ├── extracted
│           ├── raw
├── extracted_features.py
├── features.csv
├── hidden_markov_model.ipynb
└── README.md
```

- `combine_csv.py`: Script for combining raw CSV files.
- `extracted_features.py`: Feature extraction script.
- `hidden_markov_model.ipynb`: Jupyter notebook for HMM training and evaluation.
- `features.csv`: Extracted features dataset.
- `combined_activity_visualization_same_scale.png`: Visualization of combined activities.
- `data/`: Directory containing raw and processed data.

## Dependencies and Requirements
- **Python**
- **Key Libraries**:
  - `numpy`, `pandas`: Data manipulation.
  - `scipy`: FFT and signal processing.
  - `scikit-learn`: Scaling and metrics.
  - `matplotlib`, `seaborn`: Plotting.
  - `hmmlearn`: HMM implementation.
  - `joblib`: Model serialization.
## Troubleshooting
- **Empty Merged Files**: Check timestamp units (ms vs. s vs. ns). Normalize if needed (e.g., divide by 1000 for ms to s).
- **Missing session_id**: Run helper scripts to infer from gaps or chunks.
- **HMM Convergence Issues**: Increase `n_iter` or switch to `covariance_type='diag'`.
- **Feature Scaling Errors**: Ensure scaler is fitted on training only.
- **Low Accuracy**: Add more sessions or tweak window size.



## Collaboration and Contributions
- **Nicolle Nyasha Marizani**: Repository setup, data extraction/combination, initial HMM with hmmlearn.
- **Omar Keita**: Data standardization, sequence building, from-scratch HMM implementation, evaluation, and interpretation.

Part of ML Techniques 2 Formative Assessment 2 coursework.

## References
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
- For HAR specifics, refer to related works in wearable computing conferences.

```
