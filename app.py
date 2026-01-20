# =============================================================================
# app.py — Streamlit Frontend (PyTorch U-Net, OpenCV-safe)
# =============================================================================

import streamlit as st
import sys
import os
import io
import numpy as np
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Project imports (PyTorch-only)
from utils.inference import OilSpillDetector


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="🛰️ Oil Spill Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# STYLES
# =============================================================================

def inject_css():
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            padding: 2rem;
        }
        .card {
            background: rgba(255,255,255,0.95);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        .title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #0f172a;
            text-align: center;
        }
        .subtitle {
            text-align: center;
            color: #475569;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model():
    return OilSpillDetector()


# =============================================================================
# UI COMPONENTS
# =============================================================================

def header():
    st.markdown(
        """
        <div class="card">
            <div class="title">🛰️ Oil Spill Detection System</div>
            <p class="subtitle">
                AI-powered oil spill segmentation using satellite imagery
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    inject_css()
    header()

    st.markdown("### 📤 Upload Satellite Image")

    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("👆 Upload a satellite image to begin analysis.")
        return

    # 🔒 READ BYTES EXACTLY ONCE
    image_bytes = uploaded_file.read()

    if not image_bytes:
        st.error("❌ Uploaded file is empty.")
        st.stop()

    # Display original image from bytes
    display_image = Image.open(io.BytesIO(image_bytes))
    st.image(display_image, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Run Detection", type="primary"):
        with st.spinner("🔄 Loading model..."):
            detector = load_model()

        with st.spinner("🔍 Running inference..."):
            try:
                result = detector.predict(image_bytes)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.stop()

        # ================= RESULTS =================

        st.markdown("---")
        st.markdown("## 📊 Detection Results")

        metrics = result["metrics"]
        
        # Metrics display
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Coverage", f"{metrics['coverage_percentage']:.2f}%")
        with col_m2:
            st.metric("Detected Pixels", f"{metrics['detected_pixels']:,}")
        with col_m3:
            st.metric("Avg Confidence", f"{metrics['avg_confidence']:.1%}")

        # Images display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(result["original_image"], use_container_width=True)
        
        with col2:
            st.markdown("**Binary Mask**")
            st.image(result["binary_mask"], use_container_width=True, clamp=True)

        with col3:
            st.markdown("**Confidence Map**")
            st.image(result["confidence_map"], use_container_width=True, clamp=True)

        if metrics["has_spill"]:
            st.warning("⚠️ Oil spill detected!")
        else:
            st.success("✅ No oil spill detected")
      
        st.success("✅ Analysis completed successfully!")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
