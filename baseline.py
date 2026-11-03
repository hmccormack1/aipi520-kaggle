# Baseline for the Kaggle competition: will a patron buy a 2014-15 subscription?
# Style: one step per line, comments on every step, explicit loops instead of sklearn shortcuts.

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---------- STEP 1: load the files ----------
folder = "for_students/"
train = pd.read_csv(folder + "train.csv")                              # columns: account.id, label
test = pd.read_csv(folder + "test.csv")                                # column: ID
account = pd.read_csv(folder + "account.csv", encoding="latin-1")      # one row per patron
subs = pd.read_csv(folder + "subscriptions.csv", encoding="latin-1")   # one row per past subscription
tickets = pd.read_csv(folder + "tickets_all.csv", encoding="latin-1")  # one row per past ticket order

# make the test id column have the same name as the train one so we can treat them the same
test = test.rename(columns={"ID": "account.id"})

# ---------- STEP 2: build features, one small table per source, one row per account ----------
# All history files only contain seasons up to 2013-14, so none of this leaks the 2014-15 answer.

# --- subscription history features ---
subs["start_year"] = subs["season"].str[:4].astype(int)                # "2012-2013" -> 2012
sub_feat = subs.groupby("account.id").agg(
    n_sub_rows=("season", "count"),                # how many subscription rows the patron has
    n_sub_seasons=("season", "nunique"),           # how many different seasons they subscribed
    last_sub_year=("start_year", "max"),           # most recent season they subscribed
    first_sub_year=("start_year", "min"),          # earliest season they subscribed
    avg_sub_seats=("no.seats", "mean"),            # average seats per subscription
    avg_sub_price_level=("price.level", "mean"),   # average price level
).reset_index()
sub_feat["years_since_last_sub"] = 2014 - sub_feat["last_sub_year"]   # 0 means they subscribed last season

# did they subscribe in each of the last 3 seasons? (a strong signal for renewing)
for yr in [2013, 2012, 2011]:
    this_year = subs[subs["start_year"] == yr]["account.id"].unique()
    sub_feat["sub_in_" + str(yr)] = sub_feat["account.id"].isin(this_year).astype(int)

# --- single-ticket history features ---
tickets["start_year"] = tickets["season"].str[:4].astype(int)
tix_feat = tickets.groupby("account.id").agg(
    n_ticket_orders=("season", "count"),
    n_ticket_seats=("no.seats", "sum"),
    last_ticket_year=("start_year", "max"),
).reset_index()

# --- account / donation features ---
acc = account[["account.id", "amount.donated.2013", "amount.donated.lifetime", "no.donations.lifetime"]].copy()
acc["has_donated"] = (acc["no.donations.lifetime"] > 0).astype(int)

# ---------- STEP 3: attach the features to train and test ----------
def add_features(base):
    out = base.merge(acc, on="account.id", how="left")
    out = out.merge(sub_feat, on="account.id", how="left")
    out = out.merge(tix_feat, on="account.id", how="left")
    return out

train_full = add_features(train)
test_full = add_features(test)

# accounts with no history get NaN after the merge; fill with 0 (meaning "no history")
# years_since_last_sub is special: no history = very long ago, so fill with 30
feature_cols = [c for c in train_full.columns if c not in ["account.id", "label"]]
train_full["years_since_last_sub"] = train_full["years_since_last_sub"].fillna(30)
test_full["years_since_last_sub"] = test_full["years_since_last_sub"].fillna(30)
train_full[feature_cols] = train_full[feature_cols].fillna(0)
test_full[feature_cols] = test_full[feature_cols].fillna(0)

X = train_full[feature_cols]
y = train_full["label"]

# ---------- STEP 4: validation approach ----------
# hold out 25% of the labeled patrons; stratify keeps the same share of buyers (~5%) in both parts
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# ---------- STEP 5: try a few hyperparameter settings with a plain loop ----------
best_auc = 0
best_settings = None
for n_trees in [100, 200]:
    for depth in [2, 3]:
        for lr in [0.05, 0.1]:
            model = GradientBoostingClassifier(n_estimators=n_trees, max_depth=depth, learning_rate=lr, random_state=0)
            model.fit(X_tr, y_tr)                              # learn from the training part
            probs = model.predict_proba(X_val)[:, 1]           # predicted probability of buying
            auc = roc_auc_score(y_val, probs)                  # score on the held-out part
            print("trees", n_trees, "depth", depth, "lr", lr, "-> validation AUROC", round(auc, 4))
            if auc > best_auc:
                best_auc = auc
                best_settings = (n_trees, depth, lr)
print("best:", best_settings, round(best_auc, 4))

# ---------- STEP 6: refit on ALL labeled data with the best settings and write the submission ----------
final = GradientBoostingClassifier(n_estimators=best_settings[0], max_depth=best_settings[1],
                                   learning_rate=best_settings[2], random_state=0)
final.fit(X, y)
test_probs = final.predict_proba(test_full[feature_cols])[:, 1]   # soft predictions, NOT 0/1

submission = pd.DataFrame({"ID": test["account.id"], "Predicted": test_probs})
submission.to_csv("submission_baseline.csv", index=False)
print(submission.head())
