"""Praat-based voice feature extraction (matches the UCI Parkinson's column definitions)."""
import numpy as np
import parselmouth
from parselmouth.praat import call

FEATURE_COLS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
]

def extract(path: str, f0min: float = 75, f0max: float = 500) -> dict:
    snd = parselmouth.Sound(path)
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    if snd.duration < 1.0:
        raise ValueError("Recording too short - hold a vowel (e.g. 'aaah') for 3+ seconds.")

    pitch = snd.to_pitch(pitch_floor=f0min, pitch_ceiling=f0max)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) < 20:
        raise ValueError("Could not detect a stable voiced signal. Try a quieter room / sustained vowel.")

    pp = call(snd, "To PointProcess (periodic, cc)", f0min, f0max)
    j = lambda cmd: call(pp, cmd, 0, 0, 0.0001, 0.02, 1.3)
    s = lambda cmd: call([snd, pp], cmd, 0, 0, 0.0001, 0.02, 1.3, 1.6)

    hnr = call(snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=f0min), "Get mean", 0, 0)
    nhr = 1.0 / (10 ** (hnr / 10))  # approximation of MDVP noise-to-harmonics ratio

    return {
        "MDVP:Fo(Hz)": float(np.mean(f0)), "MDVP:Fhi(Hz)": float(np.max(f0)), "MDVP:Flo(Hz)": float(np.min(f0)),
        "MDVP:Jitter(%)": j("Get jitter (local)"), "MDVP:Jitter(Abs)": j("Get jitter (local, absolute)"),
        "MDVP:RAP": j("Get jitter (rap)"), "MDVP:PPQ": j("Get jitter (ppq5)"), "Jitter:DDP": j("Get jitter (ddp)"),
        "MDVP:Shimmer": s("Get shimmer (local)"), "MDVP:Shimmer(dB)": s("Get shimmer (local_dB)"),
        "Shimmer:APQ3": s("Get shimmer (apq3)"), "Shimmer:APQ5": s("Get shimmer (apq5)"),
        "MDVP:APQ": s("Get shimmer (apq11)"), "Shimmer:DDA": s("Get shimmer (dda)"),
        "NHR": float(nhr), "HNR": float(hnr),
    }
