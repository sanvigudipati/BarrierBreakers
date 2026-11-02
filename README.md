# BarrierBreakers: Parkinson's Voice Screening Prototype

A research prototype that turns a short sustained "aaah" into a **voice-pattern score**, built for the Break the Barrier healthcare competition.

> **Screening aid only. This is not a medical diagnosis and has not been clinically validated.**

## What it does
1. **Record or upload** a ~5-second sustained "aaah" (phone or laptop microphone).
2. **Extract** 16 acoustic measures with Praat (via `praat-parselmouth`): pitch (Fo, Fhi, Flo), jitter, shimmer, noise-to-harmonics ratio, and HNR.
3. **Score**: a model returns a non-diagnostic voice-pattern score, with guidance (typical / inconclusive / follow up with a clinician).

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m streamlit run app.py
```
`model_figshare.joblib` is included, so the app runs without retraining.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `features.py` | Praat feature extraction (16 measures) |
| `train.py` | Original model trained on the UCI Parkinson's dataset |
| `batch_test.py` | Tests the UCI model on real phone recordings |
| `train_figshare.py` | Trains on the Figshare phone recordings (the deployed model) |
| `age_check.py` | Age-confound and age-matched analysis |
| `features2.py`, `train_v2.py` | Expanded features (CPPS, MFCC), age-normed models, nested cross-validation |

## Results (honest summary)
Data: 81 phone recordings (40 Parkinson's, 41 healthy controls).

| Experiment | AUC |
|---|---|
| UCI-trained model tested on phone recordings | 0.34 (did not transfer) |
| Retrained on phone recordings, voice only (16 features) | ~0.63 |
| Expanded features, voice only (best of 18 configurations, optimistic) | up to ~0.75 |
| Age alone | ~0.86 |
| Age + age-normed voice (nested CV) | ~0.89 |
| Voice only, age-matched subset (n = 28) | ~0.70 (preliminary) |

**Key limitation:** the Parkinson's group averaged ~67 years old vs ~48 for controls, so age is a major confound. Voice features add little beyond age in this dataset, and the age-matched subset is too small to confirm a signal. Next steps: larger age-matched cohorts, tracking each person's own voice over time, and clinical validation.

## Data
- UCI Parkinson's dataset: https://archive.ics.uci.edu/dataset/174/parkinsons
- Voice Samples for Patients with Parkinson's Disease and Healthy Controls (Prior et al., Figshare, CC BY 4.0): https://doi.org/10.6084/m9.figshare.23849127

The recordings are not included in this repository; download them from Figshare to rerun the training scripts.

## Team
Sivaranjani Subramanian, Sanvi Gudipati, Shobhita Ramasamy Sundararajan, Kasish Jain
