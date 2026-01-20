# =============================================================================
# AquaScan Intelligence — FINAL Streamlit Frontend (Version-safe)
# =============================================================================

import streamlit as st
import base64
from pathlib import Path

from utils.inference import OilSpillDetector


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="AquaScan Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# VIDEO BACKGROUND + CSS
# =============================================================================

def inject_video_background():
    base_dir = Path(__file__).parent
    video_path = base_dir / "styles" / "oilspill.mp4"

    if not video_path.exists():
        return

    video_base64 = base64.b64encode(video_path.read_bytes()).decode()

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    * {{ font-family: 'Inter', sans-serif; }}

    .video-bg {{
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        z-index: -2;
        overflow: hidden;
    }}

    .video-bg video {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        opacity: 0.35;
    }}

    .overlay {{
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: linear-gradient(
            135deg,
            rgba(1, 15, 35, 0.92),
            rgba(8, 30, 50, 0.9)
        );
        z-index: -1;
    }}

    .hero {{
        text-align: center;
        padding: 120px 20px 80px;
        max-width: 900px;
        margin: auto;
    }}

    .hero h1 {{
        font-size: 4.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #F59E0B, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .hero p {{
        font-size: 1.15rem;
        color: rgba(255,255,255,0.75);
        margin-top: 25px;
        line-height: 1.7;
    }}

    .section {{
        max-width: 1200px;
        margin: auto;
        padding: 40px 20px;
    }}

    .card {{
        background: rgba(15, 30, 50, 0.6);
        border-radius: 20px;
        padding: 40px;
        border: 1px solid rgba(8,145,178,0.3);
        backdrop-filter: blur(12px);
    }}
    </style>

    <div class="video-bg">
        <video autoplay muted loop playsinline>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
    </div>
    <div class="overlay"></div>
    """, unsafe_allow_html=True)


inject_video_background()


# =============================================================================
# HERO
# =============================================================================

st.markdown("""
<div class="hero">
    <h1>AquaScan Intelligence</h1>
    <p>
        AI-powered marine oil spill detection using deep learning segmentation.
        Upload satellite imagery to detect contamination with precision.
    </p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL (CACHED)
# =============================================================================

@st.cache_resource
def load_detector():
    return OilSpillDetector(model_path="./models/best_model.pth")


# =============================================================================
# DETECTION SECTION
# =============================================================================

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown("## 🔍 Upload & Analyze")

uploaded_file = st.file_uploader(
    "Upload satellite or aerial image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # READ BYTES ONCE
    image_bytes = uploaded_file.read()

    if not image_bytes:
        st.error("❌ Uploaded file is empty.")
        st.stop()

    st.image(image_bytes, caption="Uploaded Image", use_column_width=True)

    if st.button("🚀 Detect Oil Spill", type="primary", use_container_width=False):

        with st.spinner("🔄 Running inference..."):
            try:
                detector = load_detector()
                results = detector.predict(image_bytes)
            except Exception as e:
                st.error(f"❌ Inference failed: {e}")
                st.stop()

        metrics = results["metrics"]
        mask = results["binary_mask"]
        confidence = results["confidence_map"]
        original = results["original_image"]

        coverage = metrics["coverage_percentage"]

        # Status
        if coverage < 2:
            st.success("✅ No oil spill detected")
        elif coverage < 10:
            st.warning("⚠️ Minor oil traces detected")
        else:
            st.error("🚨 Significant oil spill detected")

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Coverage (%)", f"{coverage:.2f}")
        col2.metric("Detected Pixels", metrics["detected_pixels"])
        col3.metric("Max Confidence", f"{metrics['max_confidence']:.2f}")

        st.markdown("### 📊 Segmentation Output")
        c1, c2 = st.columns(2)
        with c1:
            st.image(mask, caption="Binary Mask", use_column_width=True)
        with c2:
            st.image(confidence, caption="Confidence Map", use_column_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div style="text-align:center; padding:60px 20px; color:rgba(255,255,255,0.6)">
    <p><strong>AquaScan Intelligence</strong> — Protecting Oceans with AI</p>
    <p>Built with Streamlit • PyTorch • U-Net</p>
</div>
""", unsafe_allow_html=True)
