"""Train on the Figshare phone recordings, using the SAME feature extractor as the app.

Usage (from the BarrierBreakers folder):
    python train_figshare.py PD_AH HC_AH
    python train_figshare.py PD_AH HC_AH Demographics_age_sex.xlsx   # optional age/sex check

Outputs:
    figshare_features.csv   features for every recording (cached, reusable)
    model_figshare.joblib   best model, trained on all recordings

Assumption: each recording is a different person (81 recordings ~ 81 participants).
If you learn that one person has several files, tell Claude so we can group them.
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract, FEATURE_COLS

warnings.filterwarnings("ignore")


def build_feature_table(pd_dir, hc_dir):
    rows, failed = [], []
    files = [(p, 1) for p in sorted(Path(pd_dir).rglob("*.wav"))] + \
            [(p, 0) for p in sorted(Path(hc_dir).rglob("*.wav"))]
    if not files:
        sys.exit("No .wav files found. Check the folder paths.")
    for i, (path, label) in enumerate(files, 1):
        try:
            rows.append({"file": path.name, "stem": path.stem, "status": label, **extract(str(path))})
        except Exception as e:
            failed.append((path.name, str(e)))
        if i % 10 == 0:
            print(f"  processed {i}/{len(files)}")
    print(f"Extracted {len(rows)} recordings; {len(failed)} failed.")
    for name, err in failed[:10]:
        print(f"  FAILED {name}: {err}")
    return pd.DataFrame(rows)


def demographics_check(df, xlsx_path):
    """Best-effort: does age differ between groups? Can age alone predict the label?"""
    try:
        demo = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"\nCould not read demographics ({e}). If needed: pip install openpyxl")
        return
    cols = {c.lower(): c for c in demo.columns}
    id_col = next((c for k, c in cols.items() if "id" in k), None)
    age_col = next((c for k, c in cols.items() if "age" in k), None)
    sex_col = next((c for k, c in cols.items() if "sex" in k), None)
    if id_col is None or age_col is None:
        print(f"\nDemographics columns not recognised: {list(demo.columns)}")
        return
    demo["_key"] = demo[id_col].astype(str).str.strip()
    merged = df.merge(demo, left_on=df["stem"].astype(str).str.strip(), right_on="_key", how="left")
    matched = merged[age_col].notna().sum()
    print(f"\nDemographics matched for {matched}/{len(df)} recordings.")
    if matched < len(df) * 0.8:
        print("Too few matched; filename does not equal Sample ID. Skipping age/sex check.")
        return
    print("Mean age by group (1 = Parkinson's):")
    print(merged.groupby("status")[age_col].agg(["mean", "std", "count"]).round(1))
    if sex_col:
        print("\nSex counts by group:")
        print(pd.crosstab(merged["status"], merged[sex_col]))
    sub = merged.dropna(subset=[age_col])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    r = cross_validate(make_pipeline(StandardScaler(), LogisticRegression()),
                       sub[[age_col]], sub["status"], cv=cv, scoring="roc_auc")
    print(f"\nAge-only baseline AUC = {r['test_score'].mean():.3f}  "
          "(if this is high, age is a confound the voice model could be learning)")


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("Usage: python train_figshare.py <PD_folder> <HC_folder> [Demographics.xlsx]")
    print("Extracting features (a minute or two)...")
    df = build_feature_table(sys.argv[1], sys.argv[2])
    df.to_csv("figshare_features.csv", index=False)

    X, y = df[FEATURE_COLS], df["status"]
    n_pd, n_hc = int(y.sum()), int((y == 0).sum())
    baseline = max(n_pd, n_hc) / len(y)
    print(f"\n{len(y)} recordings: {n_pd} Parkinson's, {n_hc} healthy. "
          f"Always-guess-majority accuracy = {baseline:.3f}")

    if len(sys.argv) == 4:
        demographics_check(df, sys.argv[3])

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
        "svm": make_pipeline(StandardScaler(), SVC(C=1, probability=True, class_weight="balanced")),
        "rf": RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=0),
        "gb": GradientBoostingClassifier(random_state=0),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    scoring = ["roc_auc", "recall", "precision", "accuracy", "balanced_accuracy"]

    print("\nRepeated 5-fold CV (5 repeats), mean +/- std across folds:")
    aucs = {}
    for name, m in models.items():
        r = cross_validate(m, X, y, cv=cv, scoring=scoring)
        aucs[name] = r["test_roc_auc"].mean()
        print(f"{name:7s} AUC={r['test_roc_auc'].mean():.3f}+/-{r['test_roc_auc'].std():.3f}  "
              f"sens={r['test_recall'].mean():.3f}  prec={r['test_precision'].mean():.3f}  "
              f"acc={r['test_accuracy'].mean():.3f}  bal_acc={r['test_balanced_accuracy'].mean():.3f}")

    best = max(aucs, key=aucs.get)
    print(f"\nBest by AUC: {best}")
    print("Note: choosing the best of four models on the same CV is slightly optimistic; "
          "report all four rows, not just the winner.")
    models[best].fit(X, y)
    joblib.dump(models[best], "model_figshare.joblib")
    print("Saved model_figshare.joblib. To use it, change app.py to load 'model_figshare.joblib'.")


if __name__ == "__main__":
    main()
