# Humming Song Recognition System

> End-to-end audio recognition system that identifies songs from 10-second humming/whistling recordings.

## Overview

This project builds a supervised learning pipeline to classify humming and whistling recordings into 8 song categories using the **MLEnd Hums and Whistles II Dataset** (400 samples).

## Key Features

- **139-dimensional multi-modal audio features**: MFCC, pitch (F0), melody contour, chroma, and temporal features
- **4 classifiers compared**: KNN, SVM, Random Forest, Gradient Boosting
- **pYIN-based pitch detection**: Captures melody contour as the core recognition signal
- **Optimized feature extraction**: pYIN computed once and shared across pitch & contour features (~50% faster)
- **Role-aware evaluation**: Separate accuracy metrics for humming vs. whistling

## Dataset

- [MLEndHWII_sample_400](https://github.com/thekmannn/MLEndHW_QHM5703_Sample/raw/main/MLEndHWII_sample_400.zip) (~1 GB)
- 8 song categories, 400+ 10-second audio recordings (hum + whistle)

**Download and extract to `MLEndHWII_sample_400/` directory before running:**
```bash
# Place extracted .wav files into MLEndHWII_sample_400/ directory
```

## Project Structure

```
humming-song-recognition/
├── humming_classifier_improved.py   # Main pipeline
├── MLEndHWII_sample_400/           # Audio files (not included in repo)
│   └── *.wav                        # 400+ wav files
├── analy/                           # Analysis outputs
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Update the data path in humming_classifier_improved.py (line 545)
# Then run:
python humming_classifier_improved.py
```

The pipeline executes 7 steps:
1. Load audio files (22050 Hz, max 10s)
2. Extract 139-dim features per file (pitch, contour, MFCC, chroma, temporal)
3. Standardize features
4. Encode labels and split train/test (80/20)
5. Train and evaluate 4 classifiers with 5-fold CV
6. Generate confusion matrix and PCA visualization
7. Export classification results to CSV

## Feature Extraction

| Feature Group | Dimensions | Method |
|---|---|---|
| Pitch (F0) | 16 | librosa.pyin + statistics + histogram |
| Melody Contour | 6 | F0 1st/2nd derivative, rising/falling ratio |
| MFCC | 65 | 13 MFCC + delta + delta2 + statistics |
| Chroma | 24 | chroma_stft mean + std (12 pitch classes × 2) |
| Temporal | 15 | ZCR, RMS, 10-segment energy |
| **Total** | **126** | |

## Output

- `confusion_matrix.png` — Confusion matrix of best model
- `pca_visualization.png` — 2D PCA scatter plot of features
- `classification_results.csv` — Per-sample predictions
- `analy/` — Additional analysis outputs

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Librosa](https://img.shields.io/badge/Librosa-0.10+-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)

## Author

**Yihong Sun (孙一弘)** - [GitHub](https://github.com/yihong97110)

## License

This project is for educational and portfolio purposes.
