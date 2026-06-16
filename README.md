# Comparative Analysis of Anomaly Detection Techniques for Low-Energy IoT Devices in Smart Home Environments

This repository contains the code for comparing four
machine-learning techniques for detecting anomalous energy consumption in
individual IoT devices.

## Overview

Four anomaly-detection techniques — **Isolation Forest**, **Autoencoder**,
**LSTM-based prediction**, and **Local Outlier Factor (LOF)** — are compared on
three genuine IoT devices of differing power levels: a smart plug, a
thermostat, and a motion sensor.

Because no public dataset combines real IoT devices, per-device power readings,
and labelled energy-consumption anomalies, controlled synthetic faults
(spikes, drops, and gradual drift) are injected into the devices' real power
signals at known locations. These known locations serve as ground truth,
enabling evaluation with precision, recall, and F1-score.

## Dataset

The study uses the **HomePulse: Smart Home IoT Sensor Dataset** (Kaggle, CC0 license):
https://www.kaggle.com/datasets/maulikgajera/homepulse-smart-home-iot-sensor-dataset

The dataset's real per-device power readings are used; its native anomaly label
(which relates to network/security events, not energy) is not used. Download the
CSV and place it in the project folder before running the scripts.

## Repository structure

| Script | Stage | Description |
|--------|-------|-------------|
| `iot_preprocess.py` | 1. Preprocessing | Selects one representative device per type and saves clean per-device CSVs |
| `iot_inject.py` | 2. Fault injection | Injects spike/drop/drift faults at known locations; saves `is_fault` ground truth |
| `iot_models_metrics.py` | 3. Detection + metrics | Runs the four techniques and computes precision/recall/F1 and confusion matrices |
| `iot_plots.py` | 4. Visualization | Generates the injected-signal figure and the F1 comparison chart |
| `sensitivity.py` | 5. Sensitivity analysis | Varies key parameters and reports their effect on F1-score |
| `sensitivity_plot.py` | 5. Sensitivity figure | Plots F1 vs. parameter values |

## How to run

Run the scripts in order:

```bash
python iot_preprocess.py
python iot_inject.py
python iot_models_metrics.py
python iot_plots.py
python sensitivity.py
python sensitivity_plot.py
```

Before running, set the folder path at the top of each script to the location
of the dataset and outputs on your machine.

## Methods and parameters

| Technique | Family | Key settings |
|-----------|--------|--------------|
| Isolation Forest | Isolation-based | contamination = 0.027 |
| Local Outlier Factor | Density-based | n_neighbors = 35, contamination = 0.027 |
| LSTM-based prediction | Prediction-based | weighted moving average, window = 12 |
| Autoencoder | Reconstruction-based | MLP (8-4-8), trained on 80% of the data |

Faults are injected at a target rate of ~2% using a fixed random seed (42) for
reproducibility, with magnitudes defined relative to each device's own scale to
keep the comparison unbiased across devices.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Main findings

- The **Autoencoder** achieved the highest F1-score on all three devices.
- Detection performance depended more on the **choice of technique** than on the
  device's power level: when faults are defined relative to each device's scale,
  low-power devices are not inherently harder to monitor than high-power ones.

## Note

Results are obtained under controlled conditions (synthetic dataset with injected
faults); validation on naturally occurring faults in real deployments is future work.