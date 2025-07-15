import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from sklearn.ensemble import RandomForestClassifier
import joblib
from PIL import Image, ImageDraw

# === Load Models ===
mobilenet = MobileNetV3Small(include_top=False, input_shape=(224, 224, 3), pooling='avg')
rf_model = joblib.load("random_forest_eye_model_aug.pkl")  # Make sure path is correct

# === MediaPipe ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Iris indices ===
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# === Utility functions ===
def laplacian_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.mean()

def is_aligned(center, ellipse_center, tolerance=25):
    return abs(center[0] - ellipse_center[0]) < tolerance and abs(center[1] - ellipse_center[1]) < tolerance

def get_prediction(crop):
    img = cv2.resize(crop, (224, 224))
    img = preprocess_input(img)
    features = mobilenet.predict(np.expand_dims(img, axis=0))
    prob = rf_model.predict_proba(features)[0]
    return prob[1], max(prob)  # good_eye_prob, confidence

def draw_tick(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.circle(overlay, (w//2, h//2), 30, (0, 255, 0), -1)
    cv2.putText(overlay, '✓', (w//2 - 10, h//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return overlay

# === Streamlit UI ===
st.title("👁️ Eye Image Capture with Quality Check")
frame_placeholder = st.empty()
status_text = st.empty()

capture_saved = False
cap = cv2.VideoCapture("http://192.168.71.13:8080/video")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    clean_frame = frame.copy()
    h, w = frame.shape[:2]
    ellipse_center = (w // 2, h // 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    selected_eye_crop = None
    good_eye_prob = 0
    confidence = 0
    message = ""

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mesh = face_landmarks.landmark
            left = [(int(mesh[i].x * w), int(mesh[i].y * h)) for i in LEFT_IRIS]
            right = [(int(mesh[i].x * w), int(mesh[i].y * h)) for i in RIGHT_IRIS]

            left_center = tuple(sum(x)//4 for x in zip(*left))
            right_center = tuple(sum(x)//4 for x in zip(*right))

            # Draw landmarks
            for pt in left + right:
                cv2.circle(frame, pt, 1, (0, 255, 0), -1)
            cv2.circle(frame, left_center, 2, (0, 0, 255), -1)
            cv2.circle(frame, right_center, 2, (0, 0, 255), -1)

            # Draw alignment ellipse
            cv2.ellipse(frame, ellipse_center, (60, 30), 0, 0, 360, (255, 0, 0), 1)

            if is_aligned(left_center, ellipse_center):
                selected_eye_crop = clean_frame[left_center[1]-50:left_center[1]+50, left_center[0]-50:left_center[0]+50]
            elif is_aligned(right_center, ellipse_center):
                selected_eye_crop = clean_frame[right_center[1]-50:right_center[1]+50, right_center[0]-50:right_center[0]+50]

    if selected_eye_crop is not None:
        if selected_eye_crop.shape[0] > 10 and selected_eye_crop.shape[1] > 10:
            blur = laplacian_score(selected_eye_crop)
            bright = brightness_score(selected_eye_crop)

            if blur < 400:
                message = "🔴 Too blurry — please focus"
            elif bright < 80:
                message = "🔅 Too dark — increase brightness"
            elif bright > 220:
                message = "🔆 Too bright — reduce lighting"
            else:
                good_eye_prob, confidence = get_prediction(selected_eye_crop)
                if good_eye_prob > 0.7 and confidence > 0.7 and not capture_saved:
                    cv2.imwrite("captured_eye.jpg", selected_eye_crop)
                    message = "✅ Good image captured!"
                    frame = draw_tick(frame)
                    capture_saved = True
                else:
                    message = f"🟡 Waiting... prob={good_eye_prob:.2f}, conf={confidence:.2f}"

    # Display in Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")
    status_text.markdown(f"**{message}**")

    if cv2.waitKey(1) & 0xFF == 27 or capture_saved:
        break

cap.release()
cv2.destroyAllWindows()
