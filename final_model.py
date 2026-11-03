# Final model for the Kaggle competition (classical-music-meets-classical-ml-fall-2026).
# Final choice: rank average of three models trained on history features:
#   (1) gradient boosting, (2) random forest, (3) logistic regression.
# (A variant that adds zip code features to the logistic regression is tested in ens2.py;
#  it did better in cross-validation but was not chosen. See README.)
# Run from this folder after putting the competition files in for_students/ (see README).
# Output: submission_final.csv

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ---------- STEP 1: load the files ----------
folder = "for_students/"
train = pd.read_csv(folder + "train.csv")                              # account.id, label
test = pd.read_csv(folder + "test.csv").rename(columns={"ID": "account.id"})
account = pd.read_csv(folder + "account.csv", encoding="latin-1")
subs = pd.read_csv(folder + "subscriptions.csv", encoding="latin-1")
tickets = pd.read_csv(folder + "tickets_all.csv", encoding="latin-1")

# ---------- STEP 2: history features (only seasons up to 2013-14, so nothing leaks the label) ----------
subs["start_year"] = subs["season"].str[:4].astype(int)
tickets["start_year"] = tickets["season"].str[:4].astype(int)

sub_feat = subs.groupby("account.id").agg(
    n_sub_rows=("season", "count"),
    n_sub_seasons=("season", "nunique"),
    last_sub_year=("start_year", "max"),
    first_sub_year=("start_year", "min"),
    avg_sub_seats=("no.seats", "mean"),
    avg_sub_price_level=("price.level", "mean"),
).reset_index()
sub_feat["years_since_last_sub"] = 2014 - sub_feat["last_sub_year"]
for yr in [2013, 2012, 2011]:
    ids = subs[subs["start_year"] == yr]["account.id"].unique()
    sub_feat["sub_in_" + str(yr)] = sub_feat["account.id"].isin(ids).astype(int)

tix_feat = tickets.groupby("account.id").agg(
    n_ticket_orders=("season", "count"),
    n_ticket_seats=("no.seats", "sum"),
    last_ticket_year=("start_year", "max"),
).reset_index()

acc = account[["account.id", "amount.donated.2013", "amount.donated.lifetime", "no.donations.lifetime"]].copy()
acc["has_donated"] = (acc["no.donations.lifetime"] > 0).astype(int)

# ---------- STEP 3: attach the features to train and test ----------
def add_features(base):
    out = base.merge(acc, on="account.id", how="left")
    out = out.merge(sub_feat, on="account.id", how="left")
    out = out.merge(tix_feat, on="account.id", how="left")
    cols = [c for c in out.columns if c not in ["account.id", "label"]]
    out["years_since_last_sub"] = out["years_since_last_sub"].fillna(30)   # no history = very long ago
    out[cols] = out[cols].fillna(0)                                         # no history = 0
    return out, cols

train_full, feature_cols = add_features(train)
test_full, _ = add_features(test)
X = train_full[feature_cols]
y = train_full["label"]
X_test = test_full[feature_cols]

# ---------- STEP 4: model 1 = gradient boosting ----------
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)
gb.fit(X, y)
p_gb = gb.predict_proba(X_test)[:, 1]

# ---------- STEP 5: model 2 = random forest ----------
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)
rf.fit(X, y)
p_rf = rf.predict_proba(X_test)[:, 1]

# ---------- STEP 6: model 3 = logistic regression (needs scaled inputs) ----------
a = np.log1p(X)                               # log squashes huge values such as donations
b = np.log1p(X_test)
mean = a.mean()                               # scale with training statistics only
std = a.std().replace(0, 1)
a = (a - mean) / std
b = (b - mean) / std
lr = LogisticRegression(C=0.1, max_iter=2000)
lr.fit(a, y)
p_lr = lr.predict_proba(b)[:, 1]

# ---------- STEP 7: average the three models by rank and write the submission ----------
def to_rank(p):
    return pd.Series(p).rank().values / len(p)    # AUROC only uses ordering, so ranks put models on one scale

blend = (to_rank(p_gb) + to_rank(p_rf) + to_rank(p_lr)) / 3
submission = pd.DataFrame({"ID": test["account.id"], "Predicted": blend})
submission.to_csv("submission_final.csv", index=False)
print("wrote submission_final.csv", submission.shape)
