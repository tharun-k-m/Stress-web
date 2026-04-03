import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import tempfile
import soundfile as sf
import subprocess
import sys

# Force install mediapipe if it's missing
try:
    import mediapipe
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe"])
    import mediapipe

import mediapipe.solutions.face_mesh as mp_face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== DEVICE =====================
device = torch.device("cpu")

# ===================== AUDIO MODEL =====================
import torchvision.models as models

# Instantiate ResNet18 locally
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(num_ftrs, 3))

# Load your pre-trained weights
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vsa_resnet18_3class_68pct.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Mel spectrogram transform
mel_transform = T.MelSpectrogram(
    sample_rate=24000,
    n_fft=1024,
    win_length=1024,
    hop_length=512,
    n_mels=128
)

labels = ["Low Stress", "Medium Stress", "High Stress"]

# ===================== RECOMMENDATIONS =====================
def get_recommendations(stress_level):
    recommendations = {
        "Low Stress": ["Maintain your current routine", "Listen to relaxing music", "Take short breaks during work"],
        "Medium Stress": ["Practice deep breathing exercises", "Take a short walk outside", "Drink water and relax for a few minutes"],
        "High Stress": ["Try guided meditation", "Perform slow breathing exercises", "Take a break and rest", "Talk to a friend or counselor"],
        "Calm": ["You are relaxed. Maintain healthy habits", "Keep a balanced routine"],
        "Moderate Stress": ["Take a short break", "Practice breathing exercises", "Stretch your body"]
    }
    return recommendations.get(stress_level, ["No recommendation available"])

# ===================== AUDIO PREDICTION =====================
def predict_voice(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    spec = mel_transform(wav)
    spec = torch.log(spec + 1e-9)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    if spec.shape[2] < 256:
        spec = F.pad(spec, (0, 256 - spec.shape[2]))
    else:
        spec = spec[:, :, :256]
    spec = spec.unsqueeze(0)

    with torch.no_grad():
        out = model(spec)
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob).item()

    stress_level = labels[pred]
    recommendations = get_recommendations(stress_level)
    return stress_level, recommendations

# ===================== VIDEO PREDICTION =====================
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

def dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def predict_video(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    scores = []
    baseline = None
    baseline_buffer = []
    calibrated = False
    ema_score = 0.0
    alpha = 0.15
    prev_nose = None
    head_motion_ema = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            def pt(i): return np.array([lm[i].x * w, lm[i].y * h])
            NOSE, JAW_B = 1, 152
            L_EYE_T, L_EYE_B = 159, 145
            R_EYE_T, R_EYE_B = 386, 374
            L_EYE_L, L_EYE_R = 33, 133
            R_EYE_L, R_EYE_R = 362, 263
            L_BROW = 70

            face_scale = dist(pt(NOSE), pt(JAW_B))
            eye_open = (dist(pt(L_EYE_T), pt(L_EYE_B)) + dist(pt(R_EYE_T), pt(R_EYE_B))) / (2 * face_scale)
            ear = (dist(pt(L_EYE_T), pt(L_EYE_B)) / dist(pt(L_EYE_L), pt(L_EYE_R)) + dist(pt(R_EYE_T), pt(R_EYE_B)) / dist(pt(R_EYE_L), pt(R_EYE_R))) / 2
            brow_dist = dist(pt(L_BROW), pt(L_EYE_T)) / face_scale

            if prev_nose is not None:
                motion = dist(pt(NOSE), prev_nose) / face_scale
                head_motion_ema = 0.2 * motion + 0.8 * head_motion_ema
            prev_nose = pt(NOSE)

            features = np.array([eye_open, ear, brow_dist, head_motion_ema])
            if not calibrated:
                baseline_buffer.append(features)
                if len(baseline_buffer) > 20:
                    baseline = np.median(baseline_buffer, axis=0)
                    calibrated = True
            else:
                dev = np.abs(features - baseline)
                raw_score = dev[0]*40 + dev[1]*25 + dev[2]*20 + dev[3]*30
                ema_score = alpha * raw_score + (1 - alpha) * ema_score
                scores.append(ema_score)

    cap.release()
    if not scores:
        return "No face detected"
    avg_score = np.mean(scores)
    if avg_score < 1.5:
        return "Calm"
    elif avg_score < 3:
        return "Moderate Stress"
    else:
        return "High Stress"
