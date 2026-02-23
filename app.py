import streamlit as st
import numpy as np
import cv2
import joblib
from pathlib import Path
import mediapipe as mp

# ----------------------------
# Paths (Cloud-safe)
# ----------------------------
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

# Change this to your bundle filename
BUNDLE_PATH = PROJECT_ROOT / "mlp_bundle_all.joblib"

# ----------------------------
# Load model bundle (cached)
# ----------------------------
@st.cache_resource
def load_bundle():
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["model"]
    classes = bundle["label_classes"]
    feature_cols = bundle["feature_cols"]
    return model, classes, feature_cols

# ----------------------------
# Create MediaPipe landmarker (cached)
# ----------------------------
@st.cache_resource
def make_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=2
    )
    return HandLandmarker.create_from_options(options)

model, classes, feature_cols = load_bundle()
landmarker = make_landmarker()

# ============================================================
# ✅ Needed utilities (copied from your notebook, minimal set)
# ============================================================

# ---- Preprocessing utilities ----
def apply_clahe_lab(img_bgr, clipLimit=2.0, tileGridSize=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def resize_long_side(img_bgr, long_side=512):
    h, w = img_bgr.shape[:2]
    scale = long_side / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

def preprocess_for_landmarker(img_bgr, do_clahe=True, long_side=512):
    """
    Matches your training-time extraction:
    CLAHE (lighting) + resize (speed/consistency) + RGB uint8 for MediaPipe
    """
    x = img_bgr
    if do_clahe:
        x = apply_clahe_lab(x)
    x = resize_long_side(x, long_side=long_side)
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)  # uint8 RGB

# ---- Landmark feature utilities ----
def bbox_area_from_landmarks(hand_lms, w, h):
    xs = np.array([lm.x for lm in hand_lms]) * w
    ys = np.array([lm.y for lm in hand_lms]) * h
    return float(max(1.0, (xs.max() - xs.min()) * (ys.max() - ys.min())))

def normalize_hand_landmarks(hand_lms):
    """
    21 landmarks -> 63 features (x,y,z), normalized:
    - translate by wrist (0)
    - scale by wrist->middle MCP (9) distance in (x,y)
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)  # (21,3)
    wrist = pts[0].copy()
    pts = pts - wrist

    ref = pts[9]  # middle MCP
    scale = float(np.linalg.norm(ref[:2]))
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale

    return pts.reshape(-1).tolist()

def zero_hand_vec():
    return [0.0] * 63

# ---- Visualization utility ----
def overlay_landmarks_points(img_rgb, result):
    """Draw landmark dots (up to 2 hands) on an RGB image."""
    out = img_rgb.copy()
    h, w = out.shape[:2]
    if result and result.hand_landmarks:
        for lm_list in result.hand_landmarks[:2]:
            for lm in lm_list:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(out, (cx, cy), 2, (255, 255, 255), -1)
    return out

# ---- Core prediction utility (from uploaded image) ----
def predict_from_bgr(img_bgr, do_clahe=True, long_side=512):
    # preprocess BEFORE landmarking
    img_rgb = preprocess_for_landmarker(img_bgr, do_clahe=do_clahe, long_side=long_side)
    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None, "no_hands_detected", img_rgb

    hands_sorted = sorted(
        result.hand_landmarks,
        key=lambda lms: bbox_area_from_landmarks(lms, w, h),
        reverse=True
    )

    h1_present = 1
    h2_present = 1 if len(hands_sorted) > 1 else 0

    h1_vec = normalize_hand_landmarks(hands_sorted[0])
    h2_vec = normalize_hand_landmarks(hands_sorted[1]) if h2_present else zero_hand_vec()

    row = {"h1_present": h1_present, "h2_present": h2_present}
    for i, v in enumerate(h1_vec):
        row[f"h1_f{i}"] = v
    for i, v in enumerate(h2_vec):
        row[f"h2_f{i}"] = v

    # IMPORTANT: feature order from saved bundle
    x = np.array([row[c] for c in feature_cols], dtype=np.float32).reshape(1, -1)

    probs = model.predict_proba(x)[0]
    idx = int(np.argmax(probs))

    pred_label = str(classes[idx])
    pred_conf = float(probs[idx])

    vis_rgb = overlay_landmarks_points(img_rgb, result)
    return {"label": pred_label, "confidence": pred_conf, "probs": probs}, None, vis_rgb

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="BSL Gesture Classifier", layout="centered")
st.title("BSL Hand Gesture Classifier (MediaPipe Landmarks + MLP)")

st.write("Upload an image. The app extracts hand landmarks and predicts the BSL class.")

do_clahe = st.checkbox("Apply CLAHE (lighting correction)", value=True)
long_side = st.slider("Resize long side (before landmarking)", 256, 1024, 512, step=64)
show_topk = st.checkbox("Show top-5 probabilities", value=True)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not decode the uploaded image.")
    else:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

        pred, err, vis_rgb = predict_from_bgr(img_bgr, do_clahe=do_clahe, long_side=long_side)

        if err:
            st.error(f"Prediction failed: {err}")
        else:
            st.success(f"Prediction: **{pred['label']}**  |  Confidence: **{pred['confidence']:.3f}**")
            st.image(vis_rgb, caption="After preprocessing + landmarks overlay", use_container_width=True)

            if show_topk:
                probs = pred["probs"]
                order = np.argsort(probs)[::-1][:5]
                st.write("Top-5 probabilities:")
                for i in order:
                    st.write(f"- {classes[i]}: {probs[i]:.4f}")
else:
    st.info("Upload an image to begin.")
