# EDA for the Kaggle competition. Run from the folder that contains for_students/.
# Each section prints a small table so you can read it and decide what to do.

import pandas as pd

folder = "for_students/"
train = pd.read_csv(folder + "train.csv")
test = pd.read_csv(folder + "test.csv").rename(columns={"ID": "account.id"})
account = pd.read_csv(folder + "account.csv", encoding="latin-1")
subs = pd.read_csv(folder + "subscriptions.csv", encoding="latin-1")
tickets = pd.read_csv(folder + "tickets_all.csv", encoding="latin-1")

# ---- 1. class balance ----
print("\n1. LABEL BALANCE")
print(train["label"].value_counts())
print("share who bought:", round(train["label"].mean(), 3))

# ---- 2. missing values in the account file ----
print("\n2. SHARE MISSING PER COLUMN (account.csv)")
print(account.isna().mean().round(2))

# ---- 3. how many patrons have each kind of history, train vs test ----
print("\n3. SHARE OF PATRONS WITH HISTORY (train vs test)")
for name, df in [("train", train), ("test", test)]:
    has_sub = df["account.id"].isin(subs["account.id"]).mean()
    has_tix = df["account.id"].isin(tickets["account.id"]).mean()
    has_acc = df["account.id"].isin(account["account.id"]).mean()
    print(name, "| has subscription history:", round(has_sub, 3),
          "| has ticket history:", round(has_tix, 3), "| in account file:", round(has_acc, 3))

# ---- 4. buy rate by group (this shows which features matter) ----
subs["start_year"] = subs["season"].str[:4].astype(int)
last_year_subs = subs[subs["start_year"] == 2013]["account.id"].unique()
n_seasons = subs.groupby("account.id")["season"].nunique()

t = train.merge(account, on="account.id", how="left")
t["subscribed_2013"] = t["account.id"].isin(last_year_subs)
t["any_subscription_history"] = t["account.id"].isin(subs["account.id"])
t["any_ticket_history"] = t["account.id"].isin(tickets["account.id"])
t["has_donated"] = t["no.donations.lifetime"].fillna(0) > 0
t["n_seasons_subscribed"] = t["account.id"].map(n_seasons).fillna(0)

print("\n4. BUY RATE BY GROUP")
for col in ["subscribed_2013", "any_subscription_history", "any_ticket_history", "has_donated"]:
    print("\n", col)
    print(t.groupby(col)["label"].agg(["mean", "count"]).round(3))

print("\n buy rate by number of seasons subscribed (capped at 10)")
t["seasons_capped"] = t["n_seasons_subscribed"].clip(upper=10)
print(t.groupby("seasons_capped")["label"].agg(["mean", "count"]).round(3))

# ---- 5. donation amounts are skewed: look at the spread ----
print("\n5. DONATION AMOUNTS (lifetime)")
print(account["amount.donated.lifetime"].describe(percentiles=[0.5, 0.9, 0.99]).round(1))

# ---- 6. zip codes: what do they look like? ----
print("\n6. BILLING ZIP CODE SAMPLE (check formats)")
print(account["billing.zip.code"].astype(str).str.len().value_counts().head())
print(account["billing.zip.code"].dropna().astype(str).sample(8, random_state=0).tolist())
