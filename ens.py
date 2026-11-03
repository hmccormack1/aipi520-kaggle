# Ensemble test: does averaging different models beat the single gradient boosting model?
# Uses the baseline features only (the extra groups did not clearly help in cv.py).
# All models are scored on the SAME folds, so the comparison is fair.
# Run from the folder that contains for_students/.

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ---------- load ----------
folder = "for_students/"
train = pd.read_csv(folder + "train.csv")
test = pd.read_csv(folder + "test.csv").rename(columns={"ID": "account.id"})
account = pd.read_csv(folder + "account.csv", encoding="latin-1")
subs = pd.read_csv(folder + "subscriptions.csv", encoding="latin-1")
tickets = pd.read_csv(folder + "tickets_all.csv", encoding="latin-1")

# ---------- baseline features (same as baseline.py) ----------
subs["start_year"] = subs["season"].str[:4].astype(int)
tickets["start_year"] = tickets["season"].str[:4].astype(int)
A = subs.groupby("account.id").agg(
    n_sub_rows=("season", "count"), n_sub_seasons=("season", "nunique"),
    last_sub_year=("start_year", "max"), first_sub_year=("start_year", "min"),
    avg_sub_seats=("no.seats", "mean"), avg_sub_price_level=("price.level", "mean")).reset_index()
A["years_since_last_sub"] = 2014 - A["last_sub_year"]
for yr in [2013, 2012, 2011]:
    ids = subs[subs["start_year"] == yr]["account.id"].unique()
    A["sub_in_" + str(yr)] = A["account.id"].isin(ids).astype(int)
tix = tickets.groupby("account.id").agg(
    n_ticket_orders=("season", "count"), n_ticket_seats=("no.seats", "sum"),
    last_ticket_year=("start_year", "max")).reset_index()
acc = account[["account.id", "amount.donated.2013", "amount.donated.lifetime", "no.donations.lifetime"]].copy()
acc["has_donated"] = (acc["no.donations.lifetime"] > 0).astype(int)

def build(base):
    out = base.merge(acc, on="account.id", how="left")
    out = out.merge(A, on="account.id", how="left")
    out = out.merge(tix, on="account.id", how="left")
    cols = [c for c in out.columns if c not in ["account.id", "label"]]
    out["years_since_last_sub"] = out["years_since_last_sub"].fillna(30)
    out[cols] = out[cols].fillna(0)
    return out, cols

train_full, feature_cols = build(train)
test_full, _ = build(test)
X = train_full[feature_cols]
y = train_full["label"]

# ---------- three different models, each returns predicted probabilities ----------
def fit_predict_gb(X_a, y_a, X_b):
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)
    model.fit(X_a, y_a)
    return model.predict_proba(X_b)[:, 1]

def fit_predict_rf(X_a, y_a, X_b):
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)
    model.fit(X_a, y_a)
    return model.predict_proba(X_b)[:, 1]

def fit_predict_lr(X_a, y_a, X_b):
    # logistic regression needs numbers on similar scales:
    # step 1: log1p squashes huge values (like donations); step 2: standardize using the training part only
    a = np.log1p(X_a)
    b = np.log1p(X_b)
    mean = a.mean()
    std = a.std().replace(0, 1)
    a = (a - mean) / std
    b = (b - mean) / std
    model = LogisticRegression(C=0.1, max_iter=2000)
    model.fit(a, y_a)
    return model.predict_proba(b)[:, 1]

def to_rank(p):
    # AUROC only cares about ordering, so averaging ranks keeps models on the same scale
    return pd.Series(p).rank().values / len(p)

# ---------- repeated cross-validation on the same folds for every model/combination ----------
names = ["GB", "RF", "LR", "GB+RF", "GB+RF+LR"]
results = {n: [] for n in names}
for repeat in range(3):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
    for tr_idx, val_idx in folds.split(X, y):
        X_a, y_a = X.iloc[tr_idx], y.iloc[tr_idx]
        X_b, y_b = X.iloc[val_idx], y.iloc[val_idx]
        p_gb = fit_predict_gb(X_a, y_a, X_b)
        p_rf = fit_predict_rf(X_a, y_a, X_b)
        p_lr = fit_predict_lr(X_a, y_a, X_b)
        results["GB"].append(roc_auc_score(y_b, p_gb))
        results["RF"].append(roc_auc_score(y_b, p_rf))
        results["LR"].append(roc_auc_score(y_b, p_lr))
        results["GB+RF"].append(roc_auc_score(y_b, to_rank(p_gb) + to_rank(p_rf)))
        results["GB+RF+LR"].append(roc_auc_score(y_b, to_rank(p_gb) + to_rank(p_rf) + to_rank(p_lr)))

print("model     | mean AUROC | std across 15 folds")
for n in names:
    print(n.ljust(9), "|", round(np.mean(results[n]), 4), "|", round(np.std(results[n]), 4))

# how often does each ensemble beat GB alone on the same fold?
gb = np.array(results["GB"])
for n in ["GB+RF", "GB+RF+LR"]:
    wins = int((np.array(results[n]) > gb).sum())
    print(n, "beats GB alone in", wins, "of 15 folds; mean difference", round(np.mean(np.array(results[n]) - gb), 4))

# ---------- fit on all labeled data and write an ensemble submission (use only if CV says it helps) ----------
Xt = test_full[feature_cols]
t_gb = fit_predict_gb(X, y, Xt)
t_rf = fit_predict_rf(X, y, Xt)
t_lr = fit_predict_lr(X, y, Xt)
blend = (to_rank(t_gb) + to_rank(t_rf) + to_rank(t_lr)) / 3     # between 0 and 1, higher = more likely to buy
pd.DataFrame({"ID": test["account.id"], "Predicted": blend}).to_csv("submission_ensemble.csv", index=False)
print("wrote submission_ensemble.csv")
