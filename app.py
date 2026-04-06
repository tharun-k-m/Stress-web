import streamlit as st
from core import predict_voice, predict_video, get_recommendations

st.set_page_config(page_title="Stress Detection", layout="centered")

st.title("🧠 Stress Detection App")
st.write("Upload audio or video to analyze stress levels.")

tab_audio, tab_video = st.tabs(["🎤 Audio Analysis", "🎥 Video Analysis"])

with tab_audio:
    uploaded_audio = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])
    if uploaded_audio:
        with st.spinner("Processing audio..."):
            res, recs = predict_voice(uploaded_audio)
            st.metric("Stress Level", res)
            for r in recs: st.write(f"- {r}")

with tab_video:
    uploaded_video = st.file_uploader("Upload Video (MP4/MOV)", type=["mp4", "mov"])
    if uploaded_video:
        with st.spinner("Processing video..."):
            res = predict_video(uploaded_video)
            st.metric("Stress Level", res)
            recs = get_recommendations(res)
            for r in recs: st.write(f"- {r}")
