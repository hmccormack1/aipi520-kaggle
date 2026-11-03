# Repeated cross-validation to compare feature groups fairly.
# Idea: instead of one train/validation split, split the labeled data 5 ways, 3 different times,
# and average the 15 AUROC scores. This is much steadier than one split with ~83 buyers.
# Run from the folder that contains for_students/.

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ---------- load ----------
folder = "for_students/"
train = pd.read_csv(folder + "train.csv")
account = pd.read_csv(folder + "account.csv", encoding="latin-1")
subs = pd.read_csv(folder + "subscriptions.csv", encoding="latin-1")
tickets = pd.read_csv(folder + "tickets_all.csv", encoding="latin-1")
zips = pd.read_csv(folder + "zipcodes.csv", encoding="latin-1")

subs["start_year"] = subs["season"].str[:4].astype(int)
tickets["start_year"] = tickets["season"].str[:4].astype(int)

# ---------- feature group A: the baseline features ----------
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

# ---------- feature group B: more subscription detail ----------
last5 = subs[subs["start_year"] >= 2009]
B = last5.groupby("account.id").agg(
    n_seasons_last5=("season", "nunique"),
    seats_last5=("no.seats", "sum")).reset_index()
B2 = subs.groupby("account.id").agg(
    n_locations=("location", "nunique"), n_packages=("package", "nunique")).reset_index()
B = B.merge(B2, on="account.id", how="outer")

# ---------- feature group C: more ticket detail ----------
C = tickets.groupby("account.id").agg(
    n_ticket_seasons=("season", "nunique"),
    n_ticket_locations=("location", "nunique"),
    n_ticket_sets=("set", "nunique"),
    n_marketing_known=("marketing.source", "count")).reset_index()

# ---------- feature group D: donation detail ----------
D = account[["account.id", "first.donated", "amount.donated.lifetime", "no.donations.lifetime"]].copy()
D["first_donated_year"] = pd.to_datetime(D["first.donated"], errors="coerce").dt.year
D["years_since_first_donation"] = 2014 - D["first_donated_year"]
D["avg_donation"] = D["amount.donated.lifetime"] / D["no.donations.lifetime"].replace(0, np.nan)
D = D[["account.id", "years_since_first_donation", "avg_donation"]]

# ---------- feature group E: zip code information ----------
z = account[["account.id", "billing.zip.code"]].copy()
z["zip5"] = z["billing.zip.code"].astype(str).str[:5]          # cut ZIP+4 down to 5 digits
z["zip5"] = z["zip5"].str.replace(".0", "", regex=False)       # remove any ".0" from number-style zips
z["zip5"] = z["zip5"].where(z["zip5"].str.isdigit(), np.nan)   # "nan" text and junk -> real missing
z["zip5"] = z["zip5"].str.zfill(5)                             # 4-digit zips get their leading zero back
z["zip_num"] = pd.to_numeric(z["zip5"], errors="coerce")
zips2 = zips[["Zipcode", "Lat", "Long", "EstimatedPopulation", "TaxReturnsFiled", "TotalWages"]].drop_duplicates("Zipcode")
E = z.merge(zips2, left_on="zip_num", right_on="Zipcode", how="left")
E["wages_per_return"] = E["TotalWages"] / E["TaxReturnsFiled"]
E = E[["account.id", "Lat", "Long", "EstimatedPopulation", "wages_per_return"]]

# ---------- put a chosen set of groups onto the train table ----------
def build(group_names):
    out = train.merge(acc, on="account.id", how="left")
    out = out.merge(A, on="account.id", how="left")
    out = out.merge(tix, on="account.id", how="left")
    extra = {"B": B, "C": C, "D": D, "E": E}
    for g in group_names:
        out = out.merge(extra[g], on="account.id", how="left")
    cols = [c for c in out.columns if c not in ["account.id", "label"]]
    out["years_since_last_sub"] = out["years_since_last_sub"].fillna(30)
    out[cols] = out[cols].fillna(-1 if "E" in group_names else 0)  # keep it simple: one fill value
    return out[cols], out["label"]

# ---------- repeated cross-validation, written as plain loops ----------
def cv_score(X, y):
    scores = []
    for repeat in range(3):                                         # 3 different shuffles
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
        for tr_idx, val_idx in folds.split(X, y):                   # 5 folds each
            model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                               learning_rate=0.05, random_state=0)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            probs = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[val_idx], probs))
    return np.mean(scores), np.std(scores)

# ---------- compare feature sets ----------
experiments = [[], ["B"], ["C"], ["D"], ["E"], ["B", "C", "D", "E"]]
print("groups added to baseline | mean AUROC | std across 15 folds")
for groups in experiments:
    X, y = build(groups)
    mean_auc, std_auc = cv_score(X, y)
    print(str(groups).ljust(24), round(mean_auc, 4), "+/-", round(std_auc, 4), "| n features:", X.shape[1])
