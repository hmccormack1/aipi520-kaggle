# Second round of candidates, listed before looking at any results: gradient boosting + logistic regression,
# logistic regression alone with different regularization strengths, and gradient boosting + a logistic
# regression that also uses zip code features.
# Rule decided in advance: adopt a candidate only if its mean CV gain over the current ensemble is >= 0.002
# AND it beats the current ensemble in >= 11 of 15 folds. All candidates use the SAME folds.
# Run from the folder that contains for_students/. Also writes submission_gb_lr_zip.csv (not the final model).

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


# ---------- zip code features (group E from cv.py), used only by one candidate ----------
zips = pd.read_csv(folder + "zipcodes.csv", encoding="latin-1")
z = account[["account.id", "billing.zip.code"]].copy()
z["zip5"] = z["billing.zip.code"].astype(str).str[:5]
z["zip5"] = z["zip5"].str.replace(".0", "", regex=False)
z["zip5"] = z["zip5"].where(z["zip5"].str.isdigit(), np.nan)
z["zip_num"] = pd.to_numeric(z["zip5"], errors="coerce")
zips2 = zips[["Zipcode", "Lat", "Long", "EstimatedPopulation", "TaxReturnsFiled", "TotalWages"]].drop_duplicates("Zipcode")
E = z.merge(zips2, left_on="zip_num", right_on="Zipcode", how="left")
E["wages_per_return"] = E["TotalWages"] / E["TaxReturnsFiled"]
E["has_zip_info"] = E["Lat"].notna().astype(int)
E = E[["account.id", "Lat", "Long", "EstimatedPopulation", "wages_per_return", "has_zip_info"]]
zip_cols = ["Lat", "Long", "EstimatedPopulation", "wages_per_return", "has_zip_info"]

train_E = train_full.merge(E, on="account.id", how="left")
test_E = test_full.merge(E, on="account.id", how="left")
X_E = train_E[feature_cols + zip_cols]
Xt_E = test_E[feature_cols + zip_cols]

# ---------- models ----------
def fit_predict_gb(X_a, y_a, X_b):
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)
    model.fit(X_a, y_a)
    return model.predict_proba(X_b)[:, 1]

def fit_predict_rf(X_a, y_a, X_b):
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)
    model.fit(X_a, y_a)
    return model.predict_proba(X_b)[:, 1]

def fit_predict_lr(X_a, y_a, X_b, C, with_zip=False):
    a = X_a[feature_cols].copy()
    b = X_b[feature_cols].copy()
    a = np.log1p(a)                                  # squash big values in the baseline columns
    b = np.log1p(b)
    if with_zip:
        for col in zip_cols:
            fill = X_a[col].median()                 # fill missing zip info with the training median
            a[col] = X_a[col].fillna(fill)
            b[col] = X_b[col].fillna(fill)
    mean = a.mean()
    std = a.std().replace(0, 1)
    a = (a - mean) / std
    b = (b - mean) / std
    model = LogisticRegression(C=C, max_iter=3000)
    model.fit(a, y_a)
    return model.predict_proba(b)[:, 1]

def to_rank(p):
    return pd.Series(p).rank().values / len(p)

# ---------- fixed list of candidates, decided before looking at results ----------
Cs = [0.03, 0.1, 0.3, 1.0]
names = ["CURRENT GB+RF+LR", "GB+LR", "LR C=0.03", "LR C=0.1", "LR C=0.3", "LR C=1", "GB+LR(+zip)"]
results = {n: [] for n in names}
for repeat in range(3):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
    for tr_idx, val_idx in folds.split(X_E, y):
        X_a, y_a = X_E.iloc[tr_idx], y.iloc[tr_idx]
        X_b, y_b = X_E.iloc[val_idx], y.iloc[val_idx]
        p_gb = fit_predict_gb(X_a[feature_cols], y_a, X_b[feature_cols])
        p_rf = fit_predict_rf(X_a[feature_cols], y_a, X_b[feature_cols])
        p_lr = {}
        for C in Cs:
            p_lr[C] = fit_predict_lr(X_a, y_a, X_b, C)
        p_lrz = fit_predict_lr(X_a, y_a, X_b, 0.1, with_zip=True)
        results["CURRENT GB+RF+LR"].append(roc_auc_score(y_b, to_rank(p_gb) + to_rank(p_rf) + to_rank(p_lr[0.1])))
        results["GB+LR"].append(roc_auc_score(y_b, to_rank(p_gb) + to_rank(p_lr[0.1])))
        for C in Cs:
            results["LR C=" + str(C).rstrip("0").rstrip(".") if C != 1.0 else "LR C=1"].append(roc_auc_score(y_b, p_lr[C]))
        results["GB+LR(+zip)"].append(roc_auc_score(y_b, to_rank(p_gb) + to_rank(p_lrz)))

cur = np.array(results["CURRENT GB+RF+LR"])
print("candidate           | mean AUROC | std    | beats current in | mean diff")
for n in names:
    r = np.array(results[n])
    print(n.ljust(19), "|", round(r.mean(), 4), "|", round(r.std(), 4), "|", str(int((r > cur).sum())).rjust(2), "of 15 folds |", round((r - cur).mean(), 4))
print("\nadopt rule: mean diff >= 0.002 AND beats current in >= 11 of 15 folds")

# ---------- final fit: GB (baseline features) + LR (baseline + zip features), rank average ----------
t_gb = fit_predict_gb(X_E[feature_cols], y, Xt_E[feature_cols])
t_lrz = fit_predict_lr(X_E, y, Xt_E, 0.1, with_zip=True)
blend = (to_rank(t_gb) + to_rank(t_lrz)) / 2
pd.DataFrame({"ID": test["account.id"], "Predicted": blend}).to_csv("submission_gb_lr_zip.csv", index=False)
print("wrote submission_gb_lr_zip.csv")
