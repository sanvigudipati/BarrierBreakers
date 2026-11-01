"""Train + evaluate. Uses SUBJECT-grouped CV: the UCI set has ~6 recordings per person,
so a naive random split leaks identity and inflates accuracy."""
import joblib, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from features import FEATURE_COLS

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(URL)
df["subject"] = df["name"].str.extract(r"(S\d+)")
X, y, g = df[FEATURE_COLS], df["status"], df["subject"]

models = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "svm": make_pipeline(StandardScaler(), SVC(C=3, probability=True, class_weight="balanced")),
    "rf": RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=0),
    "gb": GradientBoostingClassifier(random_state=0),
}
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
scores = {}
for name, m in models.items():
    r = cross_validate(m, X, y, groups=g, cv=cv, scoring=["roc_auc", "recall", "precision", "accuracy"])
    scores[name] = r["test_roc_auc"].mean()
    print(f"{name:7s} AUC={scores[name]:.3f}  recall={r['test_recall'].mean():.3f}  "
          f"precision={r['test_precision'].mean():.3f}  acc={r['test_accuracy'].mean():.3f}")

best = max(scores, key=scores.get)
print("Best:", best)
models[best].fit(X, y)
joblib.dump(models[best], "model.joblib")
