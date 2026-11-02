"""v2: richer features + age-normed features + honest (nested) evaluation.

Usage (from the BarrierBreakers folder, with features2.py next to this file):
    python train_v2.py PD_AH HC_AH Demographics_age_sex.xlsx
    add --refresh to recompute features instead of using the cached features_v2.csv

Age handling:
    raw         voice features as they are
    normed      each feature compared with healthy people of the SAME age and sex
    normed_age  normed voice features + age (tests: does voice add to age?)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features2 import ALL_COLS, BASE_COLS, EXTRA_COLS, extract2

warnings.filterwarnings("ignore")
CALIPER = 5


class Prep(BaseEstimator, TransformerMixin):
    """Select voice columns and (optionally) age-norm them. X is a DataFrame with
    the voice columns plus 'age' and 'male'. Norms are fit on healthy rows of the TRAINING fold only."""

    def __init__(self, voice_cols=None, mode="raw"):
        self.voice_cols = voice_cols
        self.mode = mode

    def _voice(self, X):
        V = X[list(self.voice_cols)].to_numpy(float)
        return np.where(np.isnan(V), self.med_, V)

    def fit(self, X, y):
        V = X[list(self.voice_cols)].to_numpy(float)
        self.med_ = np.nanmedian(V, axis=0)
        V = np.where(np.isnan(V), self.med_, V)
        if self.mode != "raw":
            A = X[["age", "male"]].to_numpy(float)
            hc = np.asarray(y) == 0
            self.reg_ = LinearRegression().fit(A[hc], V[hc])
            self.sd_ = (V[hc] - self.reg_.predict(A[hc])).std(axis=0) + 1e-9
        return self

    def transform(self, X):
        V = self._voice(X)
        if self.mode == "raw":
            return V
        A = X[["age", "male"]].to_numpy(float)
        Z = (V - self.reg_.predict(A)) / self.sd_
        if self.mode == "normed_age":
            Z = np.column_stack([Z, X["age"].to_numpy(float)])
        return Z


def make_model(name):
    if name == "logreg":
        return LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(100, min_samples_leaf=2, class_weight="balanced",
                                      random_state=0, n_jobs=-1)
    raise ValueError(name)


def make_pipe(cols, mode, model):
    return make_pipeline(Prep(cols, mode), StandardScaler(), make_model(model))


def build_table(pd_dir, hc_dir, xlsx, refresh):
    cache = Path("features_v2.csv")
    if cache.exists() and not refresh:
        print("Using cached features_v2.csv (use --refresh to recompute).")
        return pd.read_csv(cache)
    demo = pd.read_excel(xlsx)
    cols = {c.lower(): c for c in demo.columns}
    id_c = next(c for k, c in cols.items() if "id" in k)
    age_c = next(c for k, c in cols.items() if "age" in k)
    sex_c = next(c for k, c in cols.items() if "sex" in k)
    demo = demo.assign(_key=demo[id_c].astype(str).str.strip(),
                       age=demo[age_c].astype(float),
                       male=demo[sex_c].astype(str).str.upper().str.startswith("M").astype(int))
    info = demo.set_index("_key")[["age", "male"]]
    files = [(p, 1) for p in sorted(Path(pd_dir).rglob("*.wav"))] + \
            [(p, 0) for p in sorted(Path(hc_dir).rglob("*.wav"))]
    rows, failed = [], []
    for i, (p, label) in enumerate(files, 1):
        key = p.stem.strip()
        if key not in info.index:
            failed.append((p.name, "no demographics row"))
            continue
        age, male = float(info.loc[key, "age"]), int(info.loc[key, "male"])
        try:
            rows.append({"file": p.name, "status": label, "age": age, "male": male,
                         **extract2(str(p), "M" if male else "F")})
        except Exception as e:
            failed.append((p.name, str(e)))
        if i % 10 == 0:
            print(f"  processed {i}/{len(files)}", flush=True)
    print(f"Extracted {len(rows)}; skipped/failed {len(failed)}")
    for n, e in failed[:10]:
        print(f"  {n}: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def match_on_age(df):
    pdx = df[df.status == 1].sort_values("age")
    hc = df[df.status == 0].copy()
    keep = []
    for i, row in pdx.iterrows():
        if hc.empty:
            break
        d = (hc["age"] - row["age"]).abs()
        j = d.idxmin()
        if d[j] <= CALIPER:
            keep += [i, j]
            hc = hc.drop(j)
    return df.loc[keep]


def main():
    t0 = time.time()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 3:
        sys.exit("Usage: python train_v2.py <PD_folder> <HC_folder> <Demographics.xlsx> [--refresh]")
    df = build_table(args[0], args[1], args[2], "--refresh" in sys.argv).reset_index(drop=True)
    y = df["status"]
    print(f"\n{len(df)} recordings ({int(y.sum())} Parkinson's / {int((y == 0).sum())} healthy)")

    fsets = {"base": BASE_COLS, "base+extra": BASE_COLS + EXTRA_COLS, "all": ALL_COLS}
    keys = [(f, mode, model) for f in fsets
            for mode in ("raw", "normed", "normed_age") for model in ("logreg", "rf")]

    def build(key):
        f, mode, model = key
        return make_pipe(fsets[f], mode, model)

    # wiring check: every configuration really is what its name says
    for key in keys:
        p = build(key)
        assert p[0].mode == key[1], key
        assert isinstance(p[-1], LogisticRegression) == (key[2] == "logreg"), key
        assert list(p[0].voice_cols) == fsets[key[0]], key
    print(f"Wiring check passed for {len(keys)} configurations.")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
    age_pipe = make_pipeline(StandardScaler(), LogisticRegression())
    age_auc = cross_val_score(age_pipe, df[["age"]], y, cv=cv, scoring="roc_auc")
    print(f"\nBaseline  age only: AUC = {age_auc.mean():.3f} +/- {age_auc.std():.3f}")

    print("\nRunning all configurations (repeated 5-fold CV)...", flush=True)
    res = {}
    for i, key in enumerate(keys, 1):
        s = cross_val_score(build(key), df, y, cv=cv, scoring="roc_auc")
        res[key] = (s.mean(), s.std())
        print(f"  {i}/{len(keys)} done  ({time.time() - t0:.0f}s)", flush=True)
    print("\nOptimistic table (you are picking the best of 18):")
    print(f"{'features':11s} | {'age handling':10s} | {'model':6s} | AUC")
    for key, (m, sd) in sorted(res.items(), key=lambda kv: -kv[1][0]):
        print(f"{key[0]:11s} | {key[1]:10s} | {key[2]:6s} | {m:.3f} +/- {sd:.3f}")

    # ---- nested CV: honest estimate of "pick the best config, then deploy it" ----
    print("\nNested CV (config chosen inside each training fold)...", flush=True)
    outer = StratifiedKFold(5, shuffle=True, random_state=1)
    outer_scores = []
    for n, (tr, te) in enumerate(outer.split(df, y), 1):
        dtr, ytr = df.iloc[tr], y.iloc[tr]
        inner = StratifiedKFold(4, shuffle=True, random_state=0)
        scored = {k: cross_val_score(build(k), dtr, ytr, cv=inner, scoring="roc_auc").mean() for k in keys}
        pick = max(scored, key=scored.get)
        p = build(pick).fit(dtr, ytr)
        a = roc_auc_score(y.iloc[te], p.predict_proba(df.iloc[te])[:, 1])
        outer_scores.append(a)
        print(f"  outer fold {n}/5: picked {pick}, test AUC {a:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    outer_scores = np.array(outer_scores)
    print(f"NESTED CV AUC = {outer_scores.mean():.3f} +/- {outer_scores.std():.3f}   <- quote this one")

    # ---- diagnostics on the best config ----
    best = max(res, key=lambda k: res[k][0])
    print(f"\nBest configuration by CV: {' | '.join(best)}")
    oof = cross_val_predict(build(best), df, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    for lab, name in ((1, "Parkinson's"), (0, "healthy")):
        m = (y == lab).to_numpy()
        r = spearmanr(oof[m], df.loc[m, "age"])[0]
        print(f"  correlation of score with age within {name:11s}: rho = {r:+.2f}")
    print("  (large positive rho within a group = the score is still tracking age)")

    sub = match_on_age(df)
    ys = sub["status"]
    if len(sub) >= 16 and ys.value_counts().min() >= 5:
        n_sp = int(min(4, ys.value_counts().min()))
        cvm = RepeatedStratifiedKFold(n_splits=n_sp, n_repeats=10, random_state=0)
        sm = cross_val_score(build(best), sub, ys, cv=cvm, scoring="roc_auc")
        sa = cross_val_score(age_pipe, sub[["age"]], ys, cv=cvm, scoring="roc_auc")
        print(f"  Age-matched subset (n={len(sub)}): best config AUC = {sm.mean():.3f} +/- {sm.std():.3f}; "
              f"age-only = {sa.mean():.3f}")
    else:
        print("  Age-matched subset too small to evaluate.")

    final = build(best).fit(df, y)
    joblib.dump(final, "model_v2.joblib")
    meta = {"config": list(best), "voice_cols": fsets[best[0]], "mode": best[1], "model": best[2],
            "needs_age_sex": best[1] != "raw"}
    Path("model_v2_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model_v2.joblib and model_v2_meta.json   (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()