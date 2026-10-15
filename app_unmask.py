"""UnMask AI — image/video review with exploratory suspicious-region visualization."""
from pathlib import Path
import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="UnMask AI | Media Forensics", page_icon="🔎", layout="wide")

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 10% 0%, #eaf3ff 0, #f7f9fc 38%, #ffffff 100%);}
.block-container {padding-top: 2rem; max-width: 1250px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 20px; background: linear-gradient(120deg,#082b49,#1264ad); color:white; margin-bottom:1rem;}
.hero h1 {font-size:2.5rem; margin:0; color:white;} .hero p {margin:.4rem 0 0; color:#dbeafe;}
.small-note {font-size:.88rem;color:#53657a;}
</style>
<div class="hero"><h1>UnMask AI</h1><p>AI-generated & manipulated media review · Image and video analysis · Suspicious-region visualization</p></div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_existing_detector():
    """Try to load the repository's original detector; return a readable error if weights are invalid."""
    try:
        import inference_2  # existing repository model loader
        return inference_2, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

def region_map(rgb):
    """Create an exploratory false-colour artifact map, not a model explanation or thermal scan."""
    arr = np.asarray(rgb.convert("RGB") if isinstance(rgb, Image.Image) else rgb, dtype=np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    # High-frequency residual + local texture inconsistency; normalize robustly.
    smooth = cv2.GaussianBlur(bgr, (0, 0), 2.2)
    residual = cv2.absdiff(bgr, smooth).astype(np.float32).mean(axis=2)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.GaussianBlur(gray, (0, 0), 5)
    local_sq = cv2.GaussianBlur(gray * gray, (0, 0), 5)
    local_var = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0))
    residual = cv2.GaussianBlur(residual, (0, 0), 3)
    local_var = cv2.GaussianBlur(local_var, (0, 0), 3)
    def robust_norm(x):
        lo, hi = np.percentile(x, [5, 98])
        return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    score = 0.65 * robust_norm(residual) + 0.35 * robust_norm(local_var)
    score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), 4)
    heat = cv2.applyColorMap(np.uint8(np.clip(score, 0, 1) * 255), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(arr, 0.58, heat, 0.42, 0)
    return Image.fromarray(heat), Image.fromarray(overlay), score

def run_model_image(detector, pil_image):
    """Use original classifier if its checkpoint loads; don't invent a score if it doesn't."""
    arr = np.asarray(pil_image.convert("RGB"))
    tensor = detector.preprocess_img(arr)
    import torch
    with torch.no_grad():
        out = detector.img_model(tensor).detach().cpu().numpy().squeeze()
    if out.ndim == 0 or len(np.atleast_1d(out)) < 2:
        raise ValueError("Unexpected output from the repository classifier")
    vals = np.asarray(out[:2], dtype=float)
    # Original project treats index 0 as real and index 1 as fake.
    vals = np.exp(vals - np.max(vals)) if (vals.min() < 0 or vals.max() > 1 or not np.isclose(vals.sum(), 1, atol=.05)) else vals
    vals = vals / (vals.sum() + 1e-9)
    idx = int(np.argmax(vals))
    return ("Real" if idx == 0 else "Potentially manipulated"), float(vals[idx] * 100)

mode = st.sidebar.radio("Analysis mode", ["Image analysis", "Video analysis"])
st.sidebar.caption("False-colour overlay is an exploratory visual cue. It is not proof of manipulation.")
detector, detector_error = load_existing_detector()

if mode == "Image analysis":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        with st.spinner("Reviewing image regions…"):
            heat, overlay, score = region_map(image)
        left, right = st.columns(2)
        with left:
            st.subheader("Original media")
            st.image(image, use_container_width=True)
        with right:
            st.subheader("Suspicious-region view")
            st.image(overlay, use_container_width=True)
        with st.expander("View heatmap only"):
            st.image(heat, use_container_width=True)
        st.caption("Colour scale: blue/green = lower relative artifact response; yellow/red = higher response. This is not a thermal scan.")
        if detector is not None:
            try:
                label, confidence = run_model_image(detector, image)
                c1, c2 = st.columns(2)
                c1.metric("Model assessment", label)
                c2.metric("Model confidence", f"{confidence:.1f}%")
            except Exception as exc:
                st.warning(f"The repository classifier could not analyze this file: {exc}")
        else:
            st.warning("Classifier unavailable: the repository's bundled checkpoint files appear incomplete or incompatible. Region visualization works, but no real/fake prediction is being fabricated.")
            with st.expander("Model loading detail"):
                st.code(detector_error or "Unknown model loading error")
        st.info("Important: compression, sharpening, lighting, skin texture, and ordinary image detail can also create highlights. Treat the map as a clue, not proof.")
else:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded:
        suffix = Path(uploaded.name).suffix or ".mp4"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer()); tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            if total <= 0:
                st.error("Could not read frames from this video.")
            else:
                st.write(f"Duration: {total / fps:.1f} seconds" if fps > 0 else f"Frames available: {total}")
                n = min(6, total)
                indices = np.linspace(0, total - 1, n, dtype=int)
                cols = st.columns(2)
                for i, frame_idx in enumerate(indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if not ok: continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_img = Image.fromarray(frame_rgb)
                    _, overlay, _ = region_map(frame_img)
                    with cols[i % 2]:
                        st.markdown(f"**Frame {int(frame_idx) + 1}**")
                        st.image(overlay, use_container_width=True)
                cap.release()
                st.caption("Sampled frames show relative image-artifact patterns only; the current video view does not claim a frame-level fake probability.")
                st.info("Highlights can be caused by normal texture or compression. They are not proof of deepfake manipulation.")
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

st.divider()
st.markdown("<div class='small-note'>UnMask AI · Educational media-forensics prototype · Always verify results with independent evidence.</div>", unsafe_allow_html=True)
