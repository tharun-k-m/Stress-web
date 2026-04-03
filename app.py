import os
import sys
import zipfile
import streamlit as st

# 1. Unzip if needed
if not os.path.exists('pkgs') and os.path.exists('pkgs.zip'):
    with zipfile.ZipFile('pkgs.zip', 'r') as zip_ref:
        zip_ref.extractall('pkgs_unzipped') # Extract to a fresh folder

# 2. AUTO-FIND the library folder
# This searches for where 'mediapipe' actually lives in your zip
vendor_dir = None
for root, dirs, files in os.walk('pkgs_unzipped'):
    if 'mediapipe' in dirs:
        vendor_dir = os.path.abspath(root)
        break

if vendor_dir:
    sys.path.insert(0, vendor_dir)
    # Also add the parent just in case
    sys.path.insert(0, os.path.abspath('pkgs_unzipped'))
else:
    st.error("Could not find 'mediapipe' folder inside pkgs.zip. Please check your zip structure!")

# 3. Import
import mediapipe as mp
import cv2
from core import predict_voice, predict_video, get_recommendations



# --- Rest of your App Code ---
if "reloaded" not in st.session_state:
    st.session_state.reloaded = True

st.set_page_config(page_title="Stress Detection", layout="centered")

st.title("🧠 Stress Detection App")
st.write("Detect stress levels from audio or video uploads.")

# ... rest of your tab code remains the same ...



# ------------------ Tabs ------------------
tab_audio, tab_video = st.tabs(["Audio Stress Detection", "Video Stress Detection"])

with tab_audio:
    uploaded_audio = st.file_uploader(
        "Upload an audio file", type=["wav", "mp3"], key="audio_uploader"
    )
    if uploaded_audio is not None:
        try:
            with st.spinner("Analyzing audio..."):
                audio_stress, audio_recs = predict_voice(uploaded_audio)

            st.subheader("🎤 Voice Analysis")
            st.write("**Stress Level:**", audio_stress)
            st.write("**Recommendations:**")
            for rec in audio_recs:
                st.write(f"- {rec}")
        except Exception as e:
            st.error(f"Audio analysis failed: {e}")

with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a video file", type=["mp4", "mov"], key="video_uploader"
    )
    if uploaded_video is not None:
        try:
            with st.spinner("Analyzing video..."):
                video_stress = predict_video(uploaded_video)
                video_recs = get_recommendations(video_stress)

            st.subheader("🎥 Video Analysis")
            st.write("**Stress Level:**", video_stress)
            st.write("**Recommendations:**")
            for rec in video_recs:
                st.write(f"- {rec}")
        except Exception as e:
            st.error(f"Video analysis failed: {e}")
