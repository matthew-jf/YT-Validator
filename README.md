# YT-Validator

This script is a **machine learning pipeline to help classify YouTube claims as true or false**. 

It trains a K-Nearest Neighbors classifier on historical claims data (with known verdicts), then applies the trained model to predict the truthfulness of new, unprocessed claims. The script processes features like video views, matching duration, and claim types, performs cross-validation to evaluate model performance, and outputs prediction scores for new claims data.


## Setup

Create python env with proper Python and dependencies versions:

```shell
conda env create -f environment.yml
conda activate YT-Validator
```

Uncompress training data:

```shell
tar -xzf data/export_all_claims_202505211438.tgz -C data 
```

## Usage 

Fits by default `./data/export_all_claims_202505211438.csv`.
Customize using `--training-data /path/to/training_claims.csv`.

```shell
`python pipeline.py --prediction-data /path/to/new_claims.csv`
```