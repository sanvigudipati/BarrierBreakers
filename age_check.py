"""Does the voice model add anything once age is accounted for?

Run AFTER train_figshare.py (it needs figshare_features.csv):
    python age_check.py Demographics_age_sex.xlsx

Three analyses, all with repeated stratified cross-validation:
  A. Full set: age only vs voice only vs age + voice
  B. Age-matched subset (1:1 nearest age, within a 5-year caliper): voice only
  C. Same matched subset: age only (should be near 0.5 if matching worked)
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS

warnings.filterwarnings("ignore")
CALIPER = 5  # years


def load(xlsx):
    df = pd.read_csv("figshare_features.csv")
    demo = pd.read_excel(xlsx)
    cols = {c.lower(): c for c in demo.columns}
    id_col = next(c for k, c in cols.items() if "id" in k)
    age_col = next(c for k, c in cols.items() if "age" in k)
    demo["_key"] = demo[id_col].astype(str).str.strip()
    df["_key"] = df["stem"].astype(str).str.strip()
    m = df.merge(demo[["_key", age_col]], on="_key", how="left").rename(columns={age_col: "age"})
    m = m.dropna(subset=["age"]).reset_index(drop=True)
    print(f"Loaded {len(m)} recordings with ages.")
    return m


def match_on_age(df):
    """Greedy 1:1 matching of each Parkinson's recording to the closest-age healthy one."""
    pdx = df[df.status == 1].sort_values("age")
    hc = df[df.status == 0].copy()
    keep = []
    for i, row in pdx.iterrows():
        if hc.empty:
            break
        diffs = (hc["age"] - row["age"]).abs()
        j = diffs.idxmin()
        if diffs[j] <= CALIPER:
            keep += [i, j]
            hc = hc.drop(j)
    return df.loc[keep]


def cv_auc(model, X, y, seed=0):
    n_splits = int(max(2, min(5, y.value_counts().min())))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=10, random_state=seed)
    s = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return s.mean(), s.std()


def report(label, auc):
    print(f"  {label:28s} AUC = {auc[0]:.3f} +/- {auc[1]:.3f}")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python age_check.py Demographics_age_sex.xlsx")
    df = load(sys.argv[1])
    y = df["status"]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    rf = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=0)

    print("\nA. FULL SET (age gap between groups is large)")
    report("age only (logreg)", cv_auc(lr, df[["age"]], y))
    report("voice only (logreg)", cv_auc(lr, df[FEATURE_COLS], y))
    report("age + voice (logreg)", cv_auc(lr, df[["age"] + FEATURE_COLS], y))
    print("  -> if age + voice is not clearly above age only, voice adds little beyond age.")

    sub = match_on_age(df)
    ys = sub["status"]
    print(f"\nB/C. AGE-MATCHED SUBSET: {len(sub)} recordings "
          f"({int(ys.sum())} Parkinson's, {int((ys == 0).sum())} healthy)")
    if len(sub) < 16 or ys.value_counts().min() < 5:
        print("  Too few matched pairs for a meaningful cross-validation. "
              "Report that matching left too little data, and say this is why more data is needed.")
        return
    print("  mean age: Parkinson's {:.1f}, healthy {:.1f}".format(
        sub[sub.status == 1].age.mean(), sub[sub.status == 0].age.mean()))
    report("age only (should be ~0.5)", cv_auc(lr, sub[["age"]], ys))
    report("voice only (logreg)", cv_auc(lr, sub[FEATURE_COLS], ys))
    report("voice only (random forest)", cv_auc(rf, sub[FEATURE_COLS], ys))
    print("\n  Interpretation: voice AUC near 0.5 here means the earlier signal was mostly age.")
    print("  Small n => wide uncertainty. Report it as an estimate, not a finding.")


if __name__ == "__main__":
    main()
