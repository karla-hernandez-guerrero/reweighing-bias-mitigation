# Reweighing for Bias Mitigation

Assignment 4 | Data Wrangling | MSc Applied Data Science  
Author: Karla Hernandez Guerrero

## Overview

This project implements and evaluates Reweighing using IBM’s AI Fairness 360 (AIF360) library. It accompanies a review of Kamiran and Calders’ (2012) paper, “Data preprocessing techniques for classification without discrimination.”

The analysis examines whether Reweighing reduces disparities in favorable-outcome rates while maintaining classification performance. It evaluates the method in a new experimental setting rather than reproducing the original paper’s numerical results.

## Datasets and sensitive attributes

The notebook uses the following datasets:

| Dataset | File | Sensitive attribute | Favorable outcome |
|---|---|---|---|
| Adult Census | `adult.csv` | `sex` | Income above 50K |
| German Credit | `german.data` | `sex`, derived from `personal_status_sex` | Good credit |
| COMPAS | `compas-scores-two-years.csv` | `race` | No recorded recidivism within two years |

Upload `adult.csv` and `german.data` to Google Colab before running the notebook. The correct COMPAS file is downloaded automatically from ProPublica’s repository if it is not already present in the Colab session.

### Why use german.data?

The notebook uses `german.data`, the original German Credit dataset, because it matches the defined column names and preserves the categorical codes used to derive `sex` from `personal_status_sex`.

The alternative, `german.data-numeric`, contains numerically encoded features and additional indicator variables. It is not used because the notebook handles categorical encoding during preprocessing. The accompanying `german.doc` file provides documentation rather than observations and is not loaded into the pipeline.

### Which COMPAS file is used?

The analysis uses ProPublica’s `compas-scores-two-years.csv`, which contains the observed two-year recidivism label required for this experiment.

The earlier assessment-only file named `compas.csv` is not used. Risk-assessment scores cannot substitute for observed recidivism labels.

Source: https://github.com/propublica/compas-analysis

## Preprocessing

The notebook:

- Standardizes column names and text categories to lowercase and removes surrounding whitespace.
- Handles missing values in the selected modeling columns.
- Uses `1` for favorable outcomes and `0` for unfavorable outcomes.
- Derives the German Credit sensitive attribute from `personal_status_sex`, then removes the combined column from model inputs.
- Applies COMPAS screening filters and restricts its group comparison to African-American and Caucasian observations.
- Excludes identifiers, risk scores, and later outcome information from COMPAS predictors.
- Standardizes numerical predictors and one-hot encodes categorical predictors using transformations fitted only on the training data.

The designated privileged/unprivileged groups are Male/Female for Adult and German Credit, and Caucasian/African-American for COMPAS. These definitions remain unchanged throughout evaluation.

The selected sensitive attribute is excluded from classifier inputs but used to calculate Reweighing weights and group-fairness metrics.

## Evaluation

Before mitigation or classifier training, the notebook calculates the following metrics using the original labels of each full cleaned dataset:

- Demographic parity difference (DP_Diff)
- Disparate impact ratio (DIR)
- Consistency

It then compares an unweighted logistic-regression baseline with a classifier trained using Reweighing weights. Both models are evaluated on the same held-out test observations.

### Experimental settings

- Train/test split: 70/30
- Stratification: jointly by sensitive group and outcome label
- Random seed: 42
- Classifier: logistic regression
- Regularization parameter: `C=1.0`
- Solver: `lbfgs`
- Maximum iterations: 3000
- Reweighing: applied only to the training data

Accuracy and balanced accuracy are reported alongside the fairness metrics.

### Metric interpretation

- DP_Diff is the favorable-outcome rate for the unprivileged group minus that for the privileged group. Values closer to zero indicate greater demographic parity.
- DIR divides the unprivileged group’s favorable-outcome rate by the privileged group’s rate. Values closer to one indicate parity.
- Consistency measures agreement with five nearest neighbors using the transformed predictor features and excluding the observation itself. Higher values indicate greater neighbor agreement.

Original-label metrics provide the assignment’s required starting reference. Comparing baseline and Reweighing predictions on the same test set more directly assesses the effect of mitigation.

## How to run in Google Colab

1. Download the notebook, `adult.csv`, and `german.data` from this repository.
2. Open the notebook in Google Colab.
3. Run the installation cell.
4. Open the Files panel in Colab’s left sidebar and select “Upload to session storage.”
5. Upload:
   - `adult.csv`
   - `german.data`
6. Run the remaining cells from top to bottom. The notebook automatically downloads `compas-scores-two-years.csv` if needed.
7. Run the table-generation cell to save the results table as a PNG and CSV.

The Adult and German Credit files must be uploaded to the active Colab session before running the data-loading cell. Storing them in GitHub does not automatically make them available in Colab.

An internet connection is required to install dependencies and download COMPAS. If necessary, upload the correct COMPAS file manually using the filename `compas-scores-two-years.csv`.

## Results

The table compares the original dataset labels with baseline and Reweighing predictions.

![Fairness and classification results](reweighing_results_table.png)

Original-label rows use the full cleaned datasets. Baseline and Reweighing rows use identical held-out test sets. Accuracy is not applicable to original-label rows.

### Main findings

- Adult: Reweighing substantially reduced the demographic-parity gap, with a small decrease in accuracy. However, considerable disparity remained.
- German Credit: The baseline already showed near-parity. Reweighing produced a small additional improvement, alongside slight decreases in accuracy and consistency.
- COMPAS: Reweighing only slightly improved demographic parity relative to the baseline. Both classifiers showed greater disparity than the original labels.
- Consistency decreased slightly after Reweighing across all three datasets, illustrating that improvements in group parity do not necessarily improve every fairness measure.

These findings are based on one train/test split and should not be interpreted as statistically established differences or guarantees of fairness.

### Saved outputs

The notebook saves results in `/content/results/`, including:

- `data_audit.csv`
- `fairness_results.csv`
- Per-dataset test-prediction CSV files
- `reweighing_results_table.png`
- `reweighing_results_table.csv`

To display the table above, upload `reweighing_results_table.png` to the same repository folder as this README. The CSV files can also be uploaded for readers who want to inspect the numerical results.

## Reference

Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems, 33*, 1–33.  
DOI: 10.1007/s10115-011-0463-8

## Resources

- AIF360: https://github.com/Trusted-AI/AIF360
- Adult Census: https://archive.ics.uci.edu/dataset/2/adult
- German Credit: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- COMPAS data and analysis: https://github.com/propublica/compas-analysis
