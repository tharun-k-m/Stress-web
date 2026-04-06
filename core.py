import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import tempfile
import soundfile as sf
import mediapipe as mp
import torchvision.models as models

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

device = torch.device("cpu")

# ===================== MODELS =====================
def load_audio_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(num_ftrs, 3))
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "vsa_resnet18_3class_68pct.pth")
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_audio_model()

mel_transform = T.MelSpectrogram(
    sample_rate=24000, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
)

labels = ["Low Stress", "Medium Stress", "High Stress"]

def get_recommendations(stress_level):
    recommendations = {
        "Low Stress": ["Maintain your current routine", "Listen to relaxing music"],
        "Medium Stress": ["Practice deep breathing", "Take a short walk"],
        "High Stress": ["Try guided meditation", "Talk to a friend"],
        "Calm": ["You are relaxed. Maintain healthy habits"],
        "Moderate Stress": ["Take a short break", "Practice breathing"]
    }
    return recommendations.get(stress_level, ["No recommendation available"])

# ===================== PREDICTION =====================
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

    os.unlink(path)
    return labels[pred], get_recommendations(labels[pred])

def predict_video(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    scores, baseline_buffer = [], []
    calibrated = False
    ema_score, head_motion_ema, alpha = 0.0, 0.0, 0.15
    prev_nose = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            def pt(i): return np.array([lm[i].x * w, lm[i].y * h])
            
            dist = lambda p1, p2: np.linalg.norm(p1 - p2)
            face_scale = dist(pt(1), pt(152))
            eye_open = (dist(pt(159), pt(145)) + dist(pt(386), pt(374))) / (2 * face_scale)
            ear = (dist(pt(159), pt(145)) / dist(pt(33), pt(133)) + dist(pt(386), pt(374)) / dist(pt(362), pt(263))) / 2
            brow_dist = dist(pt(70), pt(159)) / face_scale

            if prev_nose is not None:
                motion = dist(pt(1), prev_nose) / face_scale
                head_motion_ema = 0.2 * motion + 0.8 * head_motion_ema
            prev_nose = pt(1)

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
    os.unlink(video_path)
    
    if not scores: return "No face detected"
    avg_score = np.mean(scores)
    if avg_score < 1.5: return "Calm"
    elif avg_score < 3: return "Moderate Stress"
    else: return "High Stress"
