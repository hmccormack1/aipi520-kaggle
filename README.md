# Classical Music Meets Classical ML (Fall 2026)

This is for a Kaggle competition for AIPI 520, where the task is to build a model that predicts which previous patrons of the Philharmonia Baroque Orchestra will purchase a subscription to the 2014-15 concert season (the target is whether the patron purchased a 2014-15 season subscription (0/1 label)). Predictions are submitted as probabilities and scored with AUROC.

The final model is a rank-average ensemble of gradient boosting, a random forest, and a logistic regression, trained on subscription, ticket, and donation history features.

## Getting the data

The data is not in this repo and can be found at this link: https://www.kaggle.com/competitions/classical-music-meets-classical-ml-fall-2026/data. From the main competition page, download the data by going to the "Data" tab, clicking "Download All" to download the zip file, and unzipping it into the folder the scripts are in (the folder will be named for_students/). Make sure the for_students/ folder is next to the scripts and not that the scripts end up within this folder or some other variation:

```
for_students/
    train.csv, test.csv, sample_submission.csv,
    account.csv, subscriptions.csv, tickets_all.csv,
    concerts.csv, concerts_2014-15.csv, zipcodes.csv
eda.py
cv.py
...
```

## Setup

```
pip install -r requirements.txt
```

Tested with Python 3 and scikit-learn. Run every script from this folder.

## Scripts (in the order they were used)

| Script | What it does |
|---|---|
| `eda.py` | Exploratory checks: class balance, missing values, train vs test history coverage, buy rate by group, donation skew, zip code formats. |
| `cv.py` | Repeated cross-validation (5 folds x 3 repeats) to test whether extra feature groups improve a gradient boosting model. |
| `ens.py` | Compares single models (gradient boosting, random forest, logistic regression) with rank-average ensembles on the same folds. Writes `submission_ensemble.csv`. |
| `ens2.py` | Second round with a fixed list of candidates, including a logistic regression that also uses zip code features. Writes `submission_gb_lr_zip.csv`. |
| `final_model.py` | Builds the final ensemble from scratch and writes `submission_final.csv`. |

`final_model.py` reproduces the submission that was selected on Kaggle. Running `ens.py` or `ens2.py` prints many `RuntimeWarning` messages from the logistic regression (divide by zero / overflow in matmul). The scores are unaffected.

## Approach

- **Features:** per-patron counts and recency from subscription history (seasons subscribed, years since last subscription, subscribed in each of 2011-2013, seats, price level), ticket history, and donation history. Only seasons up to 2013-14 are used, so nothing leaks the 2014-15 label. Patrons with no history get 0 (and 30 for years since last subscription).
- **Validation:** repeated stratified 5-fold cross-validation on the 6,941 labeled patrons. The training set has only about 330 buyers and the public leaderboard uses about half of the test set, so a single split or the public score is too noisy to compare models.
- **Models:** gradient boosting (100 trees, depth 3, learning rate 0.05), random forest (300 trees), and an L2 logistic regression on log-scaled, standardized features.
- **Ensemble:** the three models' predictions are converted to ranks and averaged. AUROC only depends on ordering, so the output values are between 0 and 1 but are not calibrated probabilities.

## Results

| Model | Cross-validation AUROC |
|---|---|
| Gradient boosting baseline | 0.9633 |
| Logistic regression | 0.9663 |
| Random forest | 0.9553 |
| **Final: GB + RF + LR ensemble** | **0.9682** |

- Public leaderboard score of the final ensemble: 0.97687 (baseline: 0.97665).
- Extra feature groups (more subscription, ticket and donation detail, and zip code features added to the tree model) did not clearly help.
- A gradient boosting + logistic regression ensemble with zip code features in the logistic regression scored 0.9738 in cross-validation (public 0.97574). It was not chosen because it was picked from several candidates, so part of its advantage is likely selection luck, and its public score was slightly lower.