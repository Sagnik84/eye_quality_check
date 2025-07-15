import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import joblib

# === Load Models ===
mobilenet = MobileNetV3Small(include_top=False, input_shape=(224, 224, 3), pooling='avg')
rf_model = joblib.load("random_forest_eye_model_aug.pkl")  # ✅ replace with correct path

# === Helper functions ===
def laplacian_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.mean()

def get_prediction(img):
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    features = mobilenet.predict(np.expand_dims(img, axis=0))
    prob = rf_model.predict_proba(features)[0]
    good_eye_prob = prob[1]  # Assuming index 1 = good_eye
    return good_eye_prob

# === Iris landmarks ===
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# === Video Transformer ===
mp_face_mesh = None  # Lazy init

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.capture_saved = False
        self.tick_shown = False
        self.counter = 0
        global mp_face_mesh
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        image = frame.to_ndarray(format="bgr24")
        h, w = image.shape[:2]
        clean_frame = image.copy()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        ellipse_center = (w // 2, h // 2)
        ellipse_axes = (60, 30)

        if not self.tick_shown:
            cv2.ellipse(image, ellipse_center, ellipse_axes, 0, 0, 360, (255, 0, 0), 1)

        selected_crop = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mesh = face_landmarks.landmark
                left_pts = [(int(mesh[i].x * w), int(mesh[i].y * h)) for i in LEFT_IRIS]
                right_pts = [(int(mesh[i].x * w), int(mesh[i].y * h)) for i in RIGHT_IRIS]

                left_center = tuple(sum(x) // 4 for x in zip(*left_pts))
                right_center = tuple(sum(x) // 4 for x in zip(*right_pts))

                dx = abs(left_center[0] - ellipse_center[0])
                dy = abs(left_center[1] - ellipse_center[1])

                if dx < 25 and dy < 25:
                    selected_crop = clean_frame[left_center[1]-50:left_center[1]+50,
                                                left_center[0]-50:left_center[0]+50]
                elif abs(right_center[0] - ellipse_center[0]) < 25 and abs(right_center[1] - ellipse_center[1]) < 25:
                    selected_crop = clean_frame[right_center[1]-50:right_center[1]+50,
                                                right_center[0]-50:right_center[0]+50]

        if selected_crop is not None and selected_crop.shape[0] > 10 and selected_crop.shape[1] > 10:
            blur = laplacian_score(selected_crop)
            bright = brightness_score(selected_crop)

            # Instructions on screen
            if blur < 400:
                cv2.putText(image, "🔍 Hold still - Image too blurry", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif bright < 60:
                cv2.putText(image, "💡 Increase Brightness", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif bright > 220:
                cv2.putText(image, "💡 Decrease Brightness", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                confidence = get_prediction(selected_crop)
                if confidence > 0.7 and blur > 400:
                    if not self.capture_saved:
                        cv2.imwrite("captured_eye.jpg", selected_crop)
                        self.capture_saved = True
                        self.tick_shown = True
                        print("✅ Image Captured with Good Quality and Confidence:", confidence)
                else:
                    cv2.putText(image, "🧠 Confidence too low", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if self.tick_shown:
            cv2.circle(image, (w - 60, 60), 30, (0, 255, 0), 5)
            cv2.line(image, (w - 80, 60), (w - 65, 75), (0, 255, 0), 5)
            cv2.line(image, (w - 65, 75), (w - 40, 40), (0, 255, 0), 5)

        return image

# === Streamlit UI ===
st.title("👁️ Eye Quality Capture App")
st.markdown("This app captures a clean eye image only when:")
st.markdown("- Eye is centered inside blue ellipse")
st.markdown("- Image is **sharp** (blur > 400)")
st.markdown("- Brightness is ideal")
st.markdown("- Model is confident (**> 0.7** good_eye probability)")

webrtc_streamer(key="eye-check", video_processor_factory=VideoProcessor)
