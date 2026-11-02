"""Batch-test the trained model on the Figshare Parkinson's voice dataset.

Usage (from the BarrierBreakers folder, after running train.py):
    python batch_test.py PD_AH HC_AH

Pass the extracted PD folder first, then the HC folder. Every .wav inside
(including subfolders) is scored. Results go to results.csv.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import extract, FEATURE_COLS


def collect(folder, label):
    return [(p, label) for p in sorted(Path(folder).rglob("*.wav"))]


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python batch_test.py <PD_folder> <HC_folder>")
    pd_dir, hc_dir = sys.argv[1], sys.argv[2]
    files = collect(pd_dir, 1) + collect(hc_dir, 0)
    if not files:
        sys.exit("No .wav files found. Check the folder paths.")

    model = joblib.load("model.joblib")
    rows, failed = [], []
    for path, label in files:
        try:
            feats = extract(str(path))
            score = float(model.predict_proba(pd.DataFrame([feats])[FEATURE_COLS])[0, 1])
            rows.append({"file": path.name, "true_label": label, "score": score, **feats})
        except Exception as e:
            failed.append((path.name, label, str(e)))

    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)

    print(f"\nScored {len(df)} files; {len(failed)} failed feature extraction.")
    for name, label, err in failed[:10]:
        print(f"  FAILED [{'PD' if label else 'HC'}] {name}: {err}")

    if df["true_label"].nunique() < 2:
        sys.exit("Need both PD and HC files scored to compute metrics.")

    print("\nMean risk score:")
    print(df.groupby("true_label")["score"].mean().rename({0: "Healthy", 1: "Parkinson's"}).round(3))

    pred = (df["score"] >= 0.5).astype(int)
    tp = int(((pred == 1) & (df.true_label == 1)).sum())
    tn = int(((pred == 0) & (df.true_label == 0)).sum())
    n_pd, n_hc = int((df.true_label == 1).sum()), int((df.true_label == 0).sum())
    print(f"\nAUC:         {roc_auc_score(df.true_label, df.score):.3f}")
    print(f"Sensitivity: {tp / n_pd:.3f}  ({tp}/{n_pd} PD correctly flagged)")
    print(f"Specificity: {tn / n_hc:.3f}  ({tn}/{n_hc} HC correctly cleared)")
    print(f"Accuracy:    {(tp + tn) / len(df):.3f}")
    print("\nFull per-file results saved to results.csv")


if __name__ == "__main__":
    main()
