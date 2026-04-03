import os
import sys
import zipfile
import streamlit as st

# 1. Force Unzip to a specific folder
zip_path = 'pkgs.zip'
extract_path = 'extracted_pkgs'

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    st.success("✅ pkgs.zip found and extracted!")
else:
    st.error("❌ pkgs.zip NOT found in root directory!")

# 2. Debug: Look at the folder structure
# This will help us find the deep 'mediapipe' folder
all_files = []
for root, dirs, files in os.walk(extract_path):
    for d in dirs:
        all_files.append(os.path.join(root, d))

# 3. Find the parent folder of 'mediapipe'
vendor_dir = None
for path in all_files:
    if path.endswith('mediapipe'):
        vendor_dir = os.path.dirname(os.path.abspath(path))
        break

if vendor_dir:
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    st.write(f"📂 Found mediapipe at: `{vendor_dir}`")
else:
    st.error("⚠️ Could not find a folder named 'mediapipe' inside the zip.")
    st.write("Current folders found:", all_files)

# 4. Final attempt to import
try:
    import mediapipe as mp
    import cv2
    st.success("🚀 MediaPipe and CV2 loaded successfully!")
except Exception as e:
    st.error(f"Failed to load MediaPipe: {e}")

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
