# SEI-Based Battery Health Classifier

This project uses a Random Forest Classifier to detect degradation behavior in lithium metal batteries, focusing on features associated with SEI (Solid Electrolyte Interphase) growth.

## 📊 Features

Simulated dataset includes:
- `sei_thickness` (nm)
- `sei_growth_rate` (nm/cycle)
- `avg_ce` (Coulombic Efficiency)
- `delta_impedance` (Ohms)
- `capacity_fade_rate` (slope over time)

## ⚙️ Model

- **Classifier:** Random Forest
- **Goal:** Classify battery condition as `stable` or `unstable` based on SEI behavior
- **Output:** Classification report and feature importance bar chart

## 🧪 How to Run

```bash
python sei_rf_classifier.py
```

Make sure you have the following packages installed:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## 📈 Output

- A printed classification report in the console
- A saved `feature_importance.png` plot

## 🧠 Note

All data is simulated. This is a portfolio demonstration of machine learning for battery diagnostics.

---
Made by Achyut 🚀
