# Reweighing for Bias Mitigation

Assignment 4 | Data Wrangling | MSc Applied Data Science  
Author: Karla Hernandez Guerrero

## Overview

This project implements and evaluates Reweighing using IBM’s AI Fairness 360 (AIF360) library. It accompanies a review of Kamiran and Calders’ (2012) paper, “Data preprocessing techniques for classification without discrimination.”

The analysis examines whether Reweighing reduces disparities in favorable-outcome rates while maintaining classification performance.

## Datasets and sensitive attributes

The repository includes the three dataset files used in the notebook:

| Dataset | File | Sensitive attribute |
|---|---|---|
| Adult Census | `adult.csv` | `sex` |
| German Credit | `german.data` | `sex`, derived from `personal_status_sex` |
| COMPAS | `compas.csv` | `race` |

### Why use german.data?

The notebook uses `german.data`, the original German Credit dataset, because it matches the defined column names and preserves the categorical codes used to derive `sex` from `personal_status_sex`.

The alternative, `german.data-numeric`, contains numerically encoded features and additional indicator variables. It is not used because the notebook handles categorical encoding during preprocessing. The accompanying `german.doc` file contains documentation rather than observations and is not loaded into the pipeline.

## Evaluation

Before mitigation or classifier training, the notebook calculates the following measures using the original dataset labels:

- Demographic parity difference (DP_Diff)
- Disparate impact ratio (DIR)
- Consistency

After a stratified train/test split, Reweighing is applied only to the training data. The notebook then evaluates fairness and predictive performance using the final predictions on the held-out test set.

The preprocessing steps, group definitions, split settings, and model parameters are documented in the notebook.

## How to run in Google Colab

1. Download the notebook and the three dataset files from this repository.
2. Open the notebook in Google Colab.
3. Install the dependencies using the installation cell in the notebook.
4. In Colab’s left sidebar, open the Files panel and select “Upload to session storage.”
5. Upload all three dataset files:
   - `adult.csv`
   - `compas.csv`
   - `german.data`
6. Check that the file paths in the loading cells match the uploaded files. You can right-click each file in Colab and select “Copy path” to obtain its location.
7. Run the remaining cells from top to bottom.

The datasets must be uploaded to the active Google Colab session before running the data-loading cells. Storing them in this GitHub repository does not automatically make them available to the notebook.

## Results

See the notebook for the results tables, fairness comparisons, and discussion of classification performance.

## Reference

Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems, 33*, 1–33.  
DOI: 10.1007/s10115-011-0463-8
