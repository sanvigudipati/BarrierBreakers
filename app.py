import tempfile, joblib, pandas as pd, streamlit as st
from features import extract, FEATURE_COLS

st.set_page_config(page_title="Parkinson's Voice Screen", page_icon="🎙️")
st.title("🎙️ Parkinson's Voice Screening")
st.caption("Research prototype - NOT a medical diagnosis.")

@st.cache_resource
def load(): return joblib.load("model.joblib")

st.write("Record or upload a **sustained 'aaah'** (about 5 seconds, steady pitch, quiet room).")
audio = st.audio_input("Record") or st.file_uploader("...or upload a WAV file", type=["wav"])

if audio:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio.getvalue()); path = f.name
    try:
        feats = extract(path)
        p = float(load().predict_proba(pd.DataFrame([feats])[FEATURE_COLS])[0, 1])
        st.subheader(f"Voice-pattern risk score: {p:.0%}")
        st.progress(p)
        if p >= 0.7:   st.warning("Voice markers resemble Parkinson's samples. Consider seeing a neurologist.")
        elif p >= 0.4: st.info("Inconclusive. Re-record in a quieter setting and compare.")
        else:          st.success("Voice markers look typical.")
        with st.expander("Extracted acoustic biomarkers"):
            st.dataframe(pd.Series(feats, name="value"))
    except Exception as e:
        st.error(str(e))
st.divider()
st.caption("Screening aid only. Trained on 31 speakers (UCI); not clinically validated.")
