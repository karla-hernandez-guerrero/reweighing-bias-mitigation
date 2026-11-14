# Reweighing for Bias Mitigation

Assignment 4 | Data Wrangling | MSc Applied Data Science
Author: Karla Hernandez Guerrero

## Overview

This project implements and evaluates Reweighing using IBM’s
AI Fairness 360 (AIF360) library. It accompanies a review of
Kamiran and Calders’ (2012) paper, “Data preprocessing techniques
for classification without discrimination.”

The analysis examines whether Reweighing reduces disparities
in favorable-outcome rates while maintaining classification
performance.

## Datasets and sensitive attributes

- Adult Census: sex
- German Credit: sex
- COMPAS: race

## Evaluation

The notebook reports demographic parity difference, disparate
impact ratio, and consistency using the original dataset labels.
It then evaluates fairness and predictive performance on held-out
test predictions after applying the mitigation pipeline.

## How to run

1. Open the notebook in Google Colab.
2. Install the dependencies listed in the notebook.
3. Upload adult.csv, compas.csv, and german.data.
4. Check that the file paths match their uploaded locations.
5. Run the cells from top to bottom.

The notebook documents preprocessing, group definitions,
the train/test split, and model parameters.

## Results

See the notebook for the results tables and discussion.

## Reference

Kamiran, F., & Calders, T. (2012).
Data preprocessing techniques for classification without
discrimination. Knowledge and Information Systems, 33, 1–33.
DOI: 10.1007/s10115-011-0463-8
