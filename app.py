import streamlit as st
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from core import predict_voice, predict_video, get_recommendations

st.set_page_config(page_title="Stress Detection", layout="centered")
# ... rest of your UI code



# Standard imports (Streamlit Cloud will handle these via requirements.txt)
try:
    import mediapipe as mp
    from mediapipe.solutions import face_mesh
    import cv2
    import matplotlib.pyplot as plt
except ImportError as e:
    st.error(f"Library load failed: {e}. Check your requirements.txt and packages.txt.")

# Local imports
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
