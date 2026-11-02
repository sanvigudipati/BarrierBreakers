"""Richer voice features for a sustained vowel (v2).

Adds to the original 16 MDVP-style measures:
  * sex-specific pitch range (better jitter/shimmer reliability)
  * trimming of onset/offset (phone recordings start/stop noisily)
  * resampling to 16 kHz (phone recordings come at different rates)
  * CPPS (cepstral peak prominence, smoothed), pitch variability, voiced fraction
  * MFCC 1-12 mean and standard deviation
"""
import numpy as np
import parselmouth
from parselmouth.praat import call

BASE_COLS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
]
EXTRA_COLS = ["CPPS", "F0_SD_semitones", "voiced_fraction"]
MFCC_COLS = [f"mfcc{i}_mean" for i in range(1, 13)] + [f"mfcc{i}_sd" for i in range(1, 13)]
ALL_COLS = BASE_COLS + EXTRA_COLS + MFCC_COLS


def pitch_range(sex):
    """sex: 'M', 'F' or None."""
    if sex == "M":
        return 75.0, 300.0
    if sex == "F":
        return 100.0, 500.0
    return 75.0, 500.0


def load_clean(path):
    snd = parselmouth.Sound(path)
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    if snd.sampling_frequency != 16000:
        snd = snd.resample(16000)
    dur = snd.duration
    if dur < 1.0:
        raise ValueError("Recording too short - hold a vowel for 3+ seconds.")
    # drop first/last 0.4 s (onset and offset), keep at most the central 3 s
    if dur > 2.0:
        t0, t1 = 0.4, dur - 0.4
        mid = (t0 + t1) / 2
        if t1 - t0 > 3.0:
            t0, t1 = mid - 1.5, mid + 1.5
        snd = snd.extract_part(from_time=t0, to_time=t1, preserve_times=False)
    return snd


def extract2(path, sex=None):
    f0min, f0max = pitch_range(sex)
    snd = load_clean(path)

    pitch = snd.to_pitch(pitch_floor=f0min, pitch_ceiling=f0max)
    f0_all = pitch.selected_array["frequency"]
    f0 = f0_all[f0_all > 0]
    if len(f0) < 20:
        raise ValueError("Could not detect a stable voiced signal.")
    voiced_fraction = float(len(f0) / len(f0_all))
    f0_sd_st = float(np.std(12 * np.log2(f0 / np.mean(f0))))

    pp = call(snd, "To PointProcess (periodic, cc)", f0min, f0max)
    j = lambda cmd: call(pp, cmd, 0, 0, 0.0001, 0.02, 1.3)
    s = lambda cmd: call([snd, pp], cmd, 0, 0, 0.0001, 0.02, 1.3, 1.6)

    hnr = call(snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=f0min), "Get mean", 0, 0)
    nhr = 1.0 / (10 ** (hnr / 10))

    try:
        cep = call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        cpps = float(call(cep, "Get CPPS", False, 0.02, 0.0005, 60, 330, 0.05,
                          "Parabolic", 0.001, 0, "Straight", "Robust"))
    except Exception:
        cpps = float("nan")

    mf = snd.to_mfcc(number_of_coefficients=12).to_array()  # (12 + c0?, frames)
    mf = mf[1:13] if mf.shape[0] >= 13 else mf[:12]
    out = {
        "MDVP:Fo(Hz)": float(np.mean(f0)), "MDVP:Fhi(Hz)": float(np.max(f0)), "MDVP:Flo(Hz)": float(np.min(f0)),
        "MDVP:Jitter(%)": j("Get jitter (local)"), "MDVP:Jitter(Abs)": j("Get jitter (local, absolute)"),
        "MDVP:RAP": j("Get jitter (rap)"), "MDVP:PPQ": j("Get jitter (ppq5)"), "Jitter:DDP": j("Get jitter (ddp)"),
        "MDVP:Shimmer": s("Get shimmer (local)"), "MDVP:Shimmer(dB)": s("Get shimmer (local_dB)"),
        "Shimmer:APQ3": s("Get shimmer (apq3)"), "Shimmer:APQ5": s("Get shimmer (apq5)"),
        "MDVP:APQ": s("Get shimmer (apq11)"), "Shimmer:DDA": s("Get shimmer (dda)"),
        "NHR": float(nhr), "HNR": float(hnr),
        "CPPS": cpps, "F0_SD_semitones": f0_sd_st, "voiced_fraction": voiced_fraction,
    }
    for i in range(12):
        out[f"mfcc{i + 1}_mean"] = float(np.mean(mf[i]))
        out[f"mfcc{i + 1}_sd"] = float(np.std(mf[i]))
    return out
