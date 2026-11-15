# SECOM Semiconductor Defect Detection

Machine learning project for the Introduction to Data Science course.
It predicts which semiconductor wafers will fail, using 590 in-process sensor readings plus 3 derived features (`n_missing`, `hour`, `dayofweek`).

**Dataset:** [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) (1,567 wafers, 590 sensors, about 6.6% fails)

**Run in Google Colab:** [Open the notebook](https://colab.research.google.com/drive/1GiA9LCz9Qd5Q0r-370ZgFIp2IgFLxOK3?usp=sharing)

## Problem
Wafers are tested only at the end of manufacturing, so defects are found late and cost a lot.
The goal is to flag high-risk wafers early. Because only about 6.6% of wafers fail, accuracy is misleading (always predicting "Pass" already gives about 93%), so the project focuses on PR-AUC, Fail recall and F2, and compares every model with a majority-class baseline.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook secom_defect_detection.ipynb
```
Then choose Kernel > Restart and Run All. The random seed is fixed (42), so results are reproducible. The dataset is downloaded automatically with `ucimlrepo`, so an internet connection is needed.

## Project steps
1. Problem statement and dataset
2. Data quality checks (duplicates, data types, missing values) and feature engineering
3. Exploratory data analysis with written insights
4. Train/test split (80/20, stratified) before any fitting
5. Preprocessing pipeline: median imputation, outlier clipping, scaling, SMOTE (training folds only, to avoid data leakage)
6. Cross-validated comparison: Baseline, Logistic Regression, SVM (RBF), Random Forest
7. Hyperparameter tuning (training data only)
8. Decision threshold chosen from out-of-fold predictions (not the test set)
9. Final evaluation on the untouched test set
10. Error analysis
11. Limitations and next steps

## Results
(Paste your final comparison table here, then 1-2 lines of summary, for example the best model and its PR-AUC and recall.)

## Limitations
- Only about 21 fails in the test set, so metrics are noisy.
- A random split may be optimistic because wafers are time-ordered.
- Hyperparameters and threshold were both chosen on the training data (nested cross-validation would be more rigorous).
- Data comes from a single fab and period, so it may not generalise.

## Files
- `secom_defect_detection.ipynb`: full analysis
- `requirements.txt`: Python dependencies
- `README.md`: this file
