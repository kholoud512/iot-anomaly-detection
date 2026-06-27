# Comparative Analysis of Anomaly Detection Techniques for Low-Power IoT Devices in Smart Homes

This repository contains the code for comparing four
machine-learning techniques for detecting anomalous energy consumption in
individual IoT devices.

**Research question:** *Which machine learning technique best detects anomalous
energy consumption at the individual-device level in smart-home IoT environments?*

## Overview

Four anomaly-detection techniques — **Isolation Forest**, **Autoencoder**,
**LSTM-based prediction**, and **Local Outlier Factor (LOF)** — are compared
across **60 IoT devices** of three types (smart plugs, thermostats, and motion
sensors) spanning a range of power levels.

Because no public dataset combines real IoT devices, per-device power readings,
and labelled energy-consumption anomalies, three types of controlled fault —
**spikes, drops, and gradual drift** — are injected into the devices' real power
signals at known locations. These known locations serve as ground truth, and the
results are pooled across all devices and analysed **by fault type**.

## Main findings

- **Local Outlier Factor and Isolation Forest** achieved the highest detection
  performance; the **Autoencoder** was competitive but slightly and significantly
  behind; the **LSTM-based** method was clearly weakest. Differences were
  confirmed with **McNemar's statistical test** across 60 devices.
- **Detection difficulty is governed by the fault type, not the device.** Drop
  faults are detected almost perfectly, spike faults are detected well (except by
  the LSTM-based method), and gradual drift faults are difficult for all
  techniques.

## Dataset

The study uses the **HomePulse: Smart Home IoT Sensor Dataset** (Kaggle, CC0):
https://www.kaggle.com/datasets/maulikgajera/homepulse-smart-home-iot-sensor-dataset

Only the real per-device power readings are used; the dataset's native anomaly
label (network/security events) is not used. The dataset is **not included** in
this repository due to its size — download the CSV from the link above and place
it in the project folder before running the scripts.

## Repository structure

| Script | Stage | Description |
|--------|-------|-------------|
| `iot_preprocess.py` | Preprocessing | Filters devices by type and saves per-device CSVs |
| `iot_inject.py` | Fault injection | Injects spike/drop/drift faults at known locations; records `is_fault` and fault type |
| `iot_models_metrics.py` | Detection + metrics | Runs the four techniques and computes precision/recall/F1 |
| `sensitivity.py`, `sensitivity_plot.py` | Sensitivity analysis | Varies key parameters and plots F1 vs. parameter value |
| `iot_expanded_analysis.py` | Main analysis | Runs all 60 devices, pools results by fault type, runs McNemar's test |
| `iot_plots.py` | Figures | Generates the injected-signal figure |
| `f1_60.py`, `perfault_60.py` | Figures | Generate the 60-device F1 and per-fault-type figures |

## How to run

```bash
python iot_preprocess.py
python iot_inject.py
python iot_expanded_analysis.py     # main 60-device analysis (pooled, by fault type)
python sensitivity.py
python sensitivity_plot.py
python f1_60.py
python perfault_60.py
```

Set the dataset/folder path at the top of each script to its location on your machine.

## Methods and parameters

| Technique | Family | Key settings |
|-----------|--------|--------------|
| Isolation Forest | Isolation-based | contamination = 0.027 |
| Local Outlier Factor | Density-based | n_neighbors = 35, contamination = 0.027 |
| Autoencoder | Reconstruction-based | MLP (8-4-8), trained on 80% of the data |
| LSTM-based prediction | Prediction-based | weighted moving average, window = 12 |

Faults are injected at ~2% of readings using a fixed random seed (42), with
magnitudes defined relative to each device's own scale to keep the comparison
unbiased across devices.

## Requirements

- Python 3.11
- pandas, numpy, scikit-learn, scipy, matplotlib

```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

## Note

Results are obtained under controlled conditions (synthetic dataset with injected
faults); validation on naturally occurring faults in real deployments is future work.