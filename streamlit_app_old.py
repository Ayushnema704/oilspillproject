# ============================================================================
# AQUASCAN INTELLIGENCE - Clean, Minimal Streamlit Frontend
# ============================================================================

import streamlit as st
import base64
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime
import io

# Core imports for detection
from utils.inference import get_detector
from utils.visualization import create_overlay, create_confidence_heatmap, add_metrics_overlay
from utils.preprocessing import validate_image

# Database imports
from utils.db import (
    supabase,
    save_images_to_storage,
    insert_detection_data,
    fetch_all_detections
)

# Config imports
from config import config as cfg

# ------------------------ PAGE CONFIG --------------------------------------
st.set_page_config(
    page_title="AquaScan Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------ VIDEO BACKGROUND & STYLING -----------------------
def inject_video_background():
    """Inject full-screen video background with dark overlay"""
    BASE_DIR = Path(__file__).parent
    video_file = BASE_DIR / "styles" / "oilspill.mp4"
    
    if video_file.exists():
        with open(video_file, "rb") as f:
            video_bytes = f.read()
        video_base64 = base64.b64encode(video_bytes).decode()
        
        st.markdown(f"""
        <style>
            /* Import Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
            
            /* Global Styles */
            * {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }}
            
            /* Video Background */
            .video-background {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                z-index: -2;
                overflow: hidden;
            }}
            
            .video-background video {{
                position: absolute;
                top: 50%;
                left: 50%;
                min-width: 100%;
                min-height: 100%;
                transform: translate(-50%, -50%);
                opacity: 0.35;
                object-fit: cover;
            }}
            
            /* Dark Overlay with Gradient */
            .video-overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: linear-gradient(135deg, 
                    rgba(1, 15, 35, 0.92) 0%, 
                    rgba(8, 30, 50, 0.88) 50%,
                    rgba(15, 40, 65, 0.90) 100%);
                z-index: -1;
            }}
            
            /* Hero Section */
            .hero-container {{
                text-align: center;
                padding: 80px 20px 60px 20px;
                margin-bottom: 40px;
            }}
            
            .hero-title {{
                font-size: 4.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #F59E0B 0%, #0891B2 60%, #06B6D4 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 20px;
                letter-spacing: -0.02em;
                line-height: 1.1;
            }}
            
            .hero-subtitle {{
                font-size: 1.6rem;
                color: #0891B2;
                font-weight: 600;
                margin-bottom: 25px;
                letter-spacing: 0.01em;
            }}
            
            .hero-description {{
                font-size: 1.15rem;
                color: rgba(255, 255, 255, 0.75);
                max-width: 700px;
                margin: 0 auto 40px auto;
                line-height: 1.7;
            }}
            
            .cta-button {{
                display: inline-block;
                background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
                color: white;
                padding: 16px 48px;
                border-radius: 12px;
                font-size: 1.1rem;
                font-weight: 700;
                text-decoration: none;
                transition: all 0.3s ease;
                box-shadow: 0 10px 30px rgba(245, 158, 11, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.1);
            }}
            
            .cta-button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 15px 40px rgba(245, 158, 11, 0.45);
            }}
            
            /* Card Styles */
            .stContainer > div {{
                background: rgba(15, 30, 50, 0.5);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                border: 1px solid rgba(8, 145, 178, 0.2);
                padding: 30px;
            }}
            
            /* Streamlit Element Overrides */
            .stApp {{
                background: transparent;
            }}
            
            h1, h2, h3, h4, h5, h6 {{
                color: #0891B2;
                font-weight: 700;
            }}
            
            .stMarkdown {{
                color: rgba(255, 255, 255, 0.9);
            }}
            
            /* File Uploader */
            [data-testid="stFileUploader"] {{
                background: rgba(15, 30, 50, 0.6);
                border: 2px dashed rgba(8, 145, 178, 0.4);
                border-radius: 16px;
                padding: 40px;
            }}
            
            [data-testid="stFileUploader"]:hover {{
                border-color: rgba(245, 158, 11, 0.6);
            }}
            
            /* Buttons */
            .stButton > button {{
                background: linear-gradient(135deg, #0891B2 0%, #06B6D4 100%);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 32px;
                font-weight: 600;
                font-size: 1rem;
                transition: all 0.3s ease;
            }}
            
            .stButton > button:hover {{
                transform: scale(1.05);
                box-shadow: 0 10px 30px rgba(8, 145, 178, 0.4);
            }}
            
            /* Images */
            [data-testid="stImage"] {{
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            }}
            
            /* Footer */
            .footer {{
                text-align: center;
                padding: 40px 20px;
                margin-top: 80px;
                color: rgba(255, 255, 255, 0.6);
                border-top: 1px solid rgba(8, 145, 178, 0.2);
            }}
            
            .footer a {{
                color: #F59E0B;
                text-decoration: none;
                font-weight: 600;
            }}
            
            .footer a:hover {{
                color: #0891B2;
            }}
        </style>
        
        <div class="video-background">
            <video autoplay muted loop playsinline>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
        </div>
        <div class="video-overlay"></div>
        """, unsafe_allow_html=True)

inject_video_background()


# ============================================================================
# REFINED SPILL DETECTION LOGIC - Enhanced Alert System
# ============================================================================

def get_detection_status_enhanced(coverage_percentage):
    """
    Enhanced detection status with refined threshold logic
    
    Thresholds:
    - 0% to 2%: Clear Image Mode (Blue/Green)
    - 2% to 10%: Minor Traces Mode (Yellow/Amber)
    - ≥10%: Alert Mode (Red)
    
    Args:
        coverage_percentage: Float value of spill coverage
        
    Returns:
        dict: Contains status, color, icon, message, and alert_level
    """
    if coverage_percentage < 2.0:
        # Clear Image Mode (0% - 2%)
        return {
            'status': 'CLEAR',
            'color': '#10B981',  # Green
            'bg_gradient': 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
            'icon': '✅',
            'title': 'NO OIL SPILL DETECTED',
            'message': 'Water looks clean — No immediate action required',
            'alert_level': 'success',
            'recommendation': 'Continue routine monitoring of the area.'
        }
    elif 2.0 <= coverage_percentage < 10.0:
        # Minor Traces Mode (2% - 10%)
        return {
            'status': 'MINOR_TRACES',
            'color': '#F59E0B',  # Amber/Yellow
            'bg_gradient': 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)',
            'icon': '⚠️',
            'title': 'SLIGHT OIL TRACES DETECTED',
            'message': 'Minor contamination detected — Recommend re-check or continuous monitoring',
            'alert_level': 'warning',
            'recommendation': 'Schedule follow-up analysis within 24-48 hours. Monitor for changes.'
        }
    else:
        # Alert Mode (≥10%)
        return {
            'status': 'ALERT',
            'color': '#EF4444',  # Red
            'bg_gradient': 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)',
            'icon': '🚨',
            'title': 'SIGNIFICANT OIL SPILL DETECTED',
            'message': 'Major contamination detected — IMMEDIATE ATTENTION REQUIRED!',
            'alert_level': 'critical',
            'recommendation': 'Deploy cleanup crews immediately. Establish containment perimeter. Contact environmental response team.'
        }


def display_enhanced_detection_status(results, uploaded_filename):
    """
    Display enhanced detection status with refined threshold logic
    Replaces old binary detection display
    """
    coverage_pct = results['metrics']['coverage_percentage']
    status_info = get_detection_status_enhanced(coverage_pct)
    
    # Animated Alert Banner with Glassmorphism
    st.markdown(f"""
    <style>
        @keyframes pulse-border {{
            0%, 100% {{ box-shadow: 0 0 0 0 {status_info['color']}80; }}
            50% {{ box-shadow: 0 0 0 15px {status_info['color']}00; }}
        }}
        
        .enhanced-alert-banner {{
            background: {status_info['bg_gradient']};
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            margin: 40px 0;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            animation: pulse-border 2s infinite;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(255, 255, 255, 0.2);
        }}
        
        .alert-icon {{
            font-size: 4rem;
            margin-bottom: 15px;
            display: block;
            animation: bounce 1s infinite;
        }}
        
        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-10px); }}
        }}
        
        .alert-title {{
            font-family: 'Poppins', sans-serif;
            color: #ffffff;
            font-size: 2.2rem;
            font-weight: 900;
            margin: 15px 0;
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
            letter-spacing: 1px;
        }}
        
        .alert-message {{
            font-family: 'Poppins', sans-serif;
            color: #ffffff;
            font-size: 1.3rem;
            font-weight: 600;
            margin: 15px 0;
            line-height: 1.6;
        }}
        
        .coverage-display {{
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(10px);
            padding: 20px 30px;
            border-radius: 15px;
            display: inline-block;
            margin-top: 20px;
            border: 2px solid rgba(255, 255, 255, 0.3);
        }}
        
        .coverage-value {{
            font-family: 'Poppins', sans-serif;
            color: #ffffff;
            font-size: 3rem;
            font-weight: 900;
            text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.4);
        }}
        
        .coverage-label {{
            font-family: 'Poppins', sans-serif;
            color: #ffffff;
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 5px;
            opacity: 0.9;
        }}
        
        .recommendation-box {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 12px;
            margin-top: 25px;
            border: 2px solid rgba(255, 255, 255, 0.25);
        }}
        
        .recommendation-text {{
            font-family: 'Poppins', sans-serif;
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 500;
            line-height: 1.7;
            margin: 0;
        }}
    </style>
    
    <div class="enhanced-alert-banner">
        <span class="alert-icon">{status_info['icon']}</span>
        <div class="alert-title">{status_info['title']}</div>
        <div class="alert-message">{status_info['message']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional context based on alert level
    if status_info['alert_level'] == 'critical':
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.1); padding: 20px; border-radius: 12px; 
                    border-left: 5px solid #EF4444; margin: 20px 0;">
            <h4 style="color: #EF4444; margin: 0 0 10px 0; font-family: Poppins;">
                🚨 CRITICAL ALERT - Emergency Response Required
            </h4>
            <p style="color: #1E1E1E; font-size: 1rem; margin: 0; line-height: 1.6;">
                This detection indicates a significant environmental threat. Contact your local 
                environmental protection agency and deploy response teams immediately. Time is critical 
                for effective containment and cleanup operations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    elif status_info['alert_level'] == 'warning':
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); padding: 20px; border-radius: 12px; 
                    border-left: 5px solid #F59E0B; margin: 20px 0;">
            <h4 style="color: #D97706; margin: 0 0 10px 0; font-family: Poppins;">
                ⚠️ MONITORING ADVISORY
            </h4>
            <p style="color: #1E1E1E; font-size: 1rem; margin: 0; line-height: 1.6;">
                Minor oil traces detected. While not immediately critical, this warrants attention. 
                Consider conducting additional satellite passes or deploying surveillance drones to 
                confirm and track any changes in contamination levels.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ------------------------ DATABASE HELPERS ---------------------------------
def init_database():
    """Initialize the detection records database in session state"""
    if 'detection_records' not in st.session_state:
        st.session_state.detection_records = []

def add_detection_record(filename, has_spill, coverage_pct, avg_confidence, max_confidence, detected_pixels):
    """Add a new detection record to the database"""
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'filename': filename,
        'result': 'Spill Detected ✅' if has_spill else 'No Spill ❌',
        'coverage_%': round(coverage_pct, 2),
        'avg_confidence': round(avg_confidence * 100, 1),
        'max_confidence': round(max_confidence * 100, 1),
        'detected_pixels': detected_pixels
    }
    st.session_state.detection_records.insert(0, record)
    
    if len(st.session_state.detection_records) > 50:
        st.session_state.detection_records = st.session_state.detection_records[:50]

def get_records_dataframe():
    """Convert records to pandas DataFrame"""
    if not st.session_state.detection_records:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.detection_records)


# ------------------------ HELPERS & MODEL ----------------------------------
@st.cache_resource
def load_model():
    """Load the model"""
    try:
        detector = get_detector(cfg.MODEL_PATH)
        return detector
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()


def process_image(detector, uploaded_file):
    """Process uploaded image and return results"""
    try:
        image = Image.open(uploaded_file).convert('RGB')

        is_valid, message = validate_image(image)
        if not is_valid:
            st.error(f"❌ Invalid image: {message}")
            return None

        with st.spinner('🔍 Analyzing image with AI...'):
            results = detector.predict(image)

        return results
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None


def image_to_bytes(img: Image.Image, fmt="PNG"):
    """Convert PIL image to bytes for downloads"""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def ensure_uint8(img):
    """Ensure image is in uint8 format"""
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


def save_to_supabase_with_images(filename, has_spill, coverage_pct, avg_confidence, 
                                  max_confidence, detected_pixels, overlay_img, 
                                  heatmap_img, binary_mask_img):
    """
    Complete save function: metadata to database + images to storage
    """
    try:
        if supabase is None:
            print("⚠️ Supabase client not initialized")
            return False
        
        # First, upload images to storage and get URLs
        image_urls = save_images_to_storage(
            filename, overlay_img, heatmap_img, binary_mask_img
        )
        
        if image_urls is None:
            print("⚠️ Image upload failed, saving metadata without URLs")
            image_urls = {'overlay': '', 'heatmap': '', 'binary_mask': ''}
        
        # Save metadata + image URLs to database
        data = {
            'timestamp': datetime.now().isoformat(),
            'filename': str(filename),
            'has_spill': bool(has_spill),
            'coverage_percentage': float(coverage_pct),
            'avg_confidence': float(avg_confidence),
            'max_confidence': float(max_confidence),
            'detected_pixels': int(detected_pixels),
            'overlay_url': image_urls.get('overlay', ''),
            'heatmap_url': image_urls.get('heatmap', ''),
            'binary_mask_url': image_urls.get('binary_mask', '')
        }
        
        response = insert_detection_data(data, table_name="oil_detections")
        print(f"✅ Detection data saved successfully")
        return True
    
    except Exception as e:
        print(f"❌ Error in save_to_supabase_with_images: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ======================== IMAGE GALLERY FUNCTION ======================
def display_detection_image_gallery():
    """
    Display a gallery of stored detection images from Supabase - ONLY after date selection
    """
    st.markdown("""
    <div class="gallery-header">
        <h2>🖼️ Detection Image Gallery</h2>
        <p>Select a date to view stored detection images from the cloud database</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        if supabase is None:
            st.info("⚠️ Supabase database not configured. Image gallery is unavailable.")
            return
        
        with st.spinner("Loading available dates..."):
            data = fetch_all_detections("oil_detections")
        
        if not data or len(data) == 0:
            st.info("🔭 No images found in the database yet. Upload and analyze images to populate the gallery.")
            return
        
        # Filter only detections that have at least one image URL
        detections_with_images = [
            d for d in data 
            if (d.get('overlay_url') and d.get('overlay_url') != '') or 
               (d.get('heatmap_url') and d.get('heatmap_url') != '') or 
               (d.get('binary_mask_url') and d.get('binary_mask_url') != '')
        ]
        
        if not detections_with_images:
            st.warning(f"⚠️ Found {len(data)} detection records, but no images are stored.")
            return
        
        st.markdown(f'<div class="gallery-success-message">✅ Found {len(detections_with_images)} detections with images</div>', unsafe_allow_html=True)
        
        # DATE FILTER - REQUIRED
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.markdown('<h3>📅 Select Date to View Images</h3>', unsafe_allow_html=True)
        
        # Extract unique dates
        dates = sorted(set([d.get('timestamp', '')[:10] for d in detections_with_images if d.get('timestamp')]), reverse=True)
        
        # Create filter options - DATE SELECTION REQUIRED
        selected_filter = st.selectbox(
            "Choose a date:",
            options=["-- Select a Date --"] + dates,
            index=0,
            key="image_gallery_date_filter"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ONLY SHOW IMAGES IF DATE IS SELECTED
        if selected_filter == "-- Select a Date --":
            st.info("👆 Please select a date above to view detection images.")
            return
        
        # Apply date filter
        filtered_detections = [d for d in detections_with_images if d.get('timestamp', '').startswith(selected_filter)]
        
        if not filtered_detections:
            st.warning(f"No detections found for date: {selected_filter}")
            return
        
        st.info(f"📊 Showing {len(filtered_detections)} detection(s) from {selected_filter}")
        
        # Display each detection in a card
        for idx, detection in enumerate(filtered_detections):
            st.markdown(f"""
            <div class="image-gallery-card">
                <div class="gallery-header-info">
                    <div>
                        <div class="gallery-filename">📄 {detection.get('filename', 'Unknown')}</div>
                        <div class="gallery-timestamp">🕒 {detection.get('timestamp', 'Unknown')}</div>
                    </div>
                    <div>
                        <span class="gallery-status-badge {'status-detected' if detection.get('has_spill') else 'status-clean'}">
                            {'🚨 Spill Detected' if detection.get('has_spill') else '✅ Clean'}
                        </span>
                    </div>
                </div>
                <div class="gallery-metadata-box">
                    <strong>Coverage:</strong> {detection.get('coverage_percentage', 0):.2f}% | 
                    <strong>Avg Confidence:</strong> {detection.get('avg_confidence', 0):.3f} | 
                    <strong>Max Confidence:</strong> {detection.get('max_confidence', 0):.3f} | 
                    <strong>Pixels:</strong> {detection.get('detected_pixels', 0):,}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create three columns for the three image types
            col1, col2, col3 = st.columns(3)
            
            # Overlay Image
            with col1:
                overlay_url = detection.get('overlay_url', '')
                if overlay_url and overlay_url != '':
                    st.markdown("""
                    <div class="gallery-image-box">
                        <div class="gallery-image-label">Detection Overlay</div>
                    """, unsafe_allow_html=True)
                    st.image(overlay_url, use_column_width=True)
                    st.markdown(f'<a href="{overlay_url}" target="_blank" class="image-url-link">🔗 Open Full Image</a>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-image-box">⚠️ Overlay image not available</div>', unsafe_allow_html=True)
            
            # Heatmap Image
            with col2:
                heatmap_url = detection.get('heatmap_url', '')
                if heatmap_url and heatmap_url != '':
                    st.markdown("""
                    <div class="gallery-image-box">
                        <div class="gallery-image-label">Confidence Heatmap</div>
                    """, unsafe_allow_html=True)
                    st.image(heatmap_url, use_column_width=True)
                    st.markdown(f'<a href="{heatmap_url}" target="_blank" class="image-url-link">🔗 Open Full Image</a>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-image-box">⚠️ Heatmap image not available</div>', unsafe_allow_html=True)
            
            # Binary Mask Image
            with col3:
                binary_url = detection.get('binary_mask_url', '')
                if binary_url and binary_url != '':
                    st.markdown("""
                    <div class="gallery-image-box">
                        <div class="gallery-image-label">Binary Mask</div>
                    """, unsafe_allow_html=True)
                    st.image(binary_url, use_column_width=True)
                    st.markdown(f'<a href="{binary_url}" target="_blank" class="image-url-link">🔗 Open Full Image</a>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-image-box">⚠️ Binary mask image not available</div>', unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 30px 0;"></div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error loading image gallery: {str(e)}")
        print(f"Gallery error: {e}")
        import traceback
        traceback.print_exc()


# ==================== MAIN UI -----------------------------------------
def main():
    # Initialize database
    init_database()
    
    # Initialize session state
    if 'total_processed' not in st.session_state:
        st.session_state.total_processed = 0
    if 'total_detections' not in st.session_state:
        st.session_state.total_detections = 0

    # ==================== PREMIUM HERO SECTION ====================
    st.markdown("""
    <div class="hero-container">
        <div style="position: relative; z-index: 2;">
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <span style="font-size: 4rem; animation: pulseGlow 3s ease-in-out infinite;">🛰️</span>
            </div>
            <h1 class="hero-title">AquaScan Intelligence</h1>
            <h2 class="hero-subtitle">🌊 Next-Gen Marine Pollution Detection System</h2>
            <p class="hero-tagline">PRECISION DETECTION • REAL-TIME ANALYSIS • GLOBAL SCALE</p>
            <p class="hero-description">
                Powered by advanced deep learning and satellite imagery analysis, AquaScan delivers 
                instant oil spill detection with 94.57% accuracy. Our AI-driven platform combines 
                U-Net neural networks with attention mechanisms to identify marine contamination 
                patterns across diverse maritime conditions—enabling rapid response and environmental 
                protection at a global scale.
            </p>
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(8, 145, 178, 0.1); border-radius: 16px; backdrop-filter: blur(10px); border: 1px solid rgba(245, 158, 11, 0.2);">
                <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
                    🔬 <strong style="color: #f59e0b;">Developed by</strong> 
                    <a href="https://www.linkedin.com/in/ayush-nema-jec/" target="_blank" 
                       style="color: #06b6d4; text-decoration: none; font-weight: 600; transition: all 0.3s ease;">
                        Ayush Nema
                    </a>
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==================== ABOUT SECTION ====================
    st.markdown("""
    <div class="cards-container">
        <div class="info-card">
            <span class="icon">�</span>
            <h3>Precision Technology</h3>
            <p>Built on U-Net architecture with attention mechanisms, AquaScan achieves 94.57% detection accuracy. Our convolutional neural networks identify oil contamination patterns across diverse maritime conditions, weather scenarios, and water compositions with exceptional reliability.</p>
        </div>
        <div class="info-card">
            <span class="icon">⚡</span>
            <h3>Lightning-Fast Analysis</h3>
            <p>Process satellite imagery in seconds, not hours. AquaScan's optimized inference pipeline delivers real-time detection results with confidence heatmaps, coverage metrics, and actionable intelligence—enabling immediate decision-making during critical response windows.</p>
        </div>
        <div class="info-card">
            <span class="icon">🌐</span>
            <h3>Enterprise-Grade Platform</h3>
            <p>Cloud-integrated architecture with Supabase backend provides seamless data persistence, historical tracking, and multi-user collaboration. Access detection records, generate reports, and coordinate response efforts from anywhere, anytime.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==================== UPLOAD SECTION ====================
    st.markdown('<h2 class="section-title">📤 Upload & Analyze Imagery</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Upload satellite or aerial imagery to detect potential oil spills using our AI model</p>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="For best results, use high-resolution satellite or aerial imagery"
    )

        
    # ✅ NEW CODE: Limit file size to 5 MB
    MAX_FILE_SIZE_MB = 5
    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)  # Convert bytes → MB

        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"🚫 File too large! The uploaded file is {file_size_mb:.2f} MB. Please upload a file under {MAX_FILE_SIZE_MB} MB.")
            st.stop()  # Stop further execution to prevent processing
        else:
            st.success(f"✅ File uploaded successfully ({file_size_mb:.2f} MB)")

    # Controls in columns
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=cfg.CONFIDENCE_THRESHOLD,
            step=0.01,
            help="Minimum confidence level for detection"
        )
    with col2:
        overlay_alpha = st.slider(
            "Overlay Transparency",
            min_value=0.0,
            max_value=1.0,
            value=cfg.OVERLAY_ALPHA,
            step=0.05,
            help="Transparency of the detection overlay"
    )

    # Action buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        detect_button = st.button("🔍 Detect Oil Spill", type="primary", use_container_width=True)
    with col_btn2:
        clear_button = st.button("♻️ Clear Results", use_container_width=True)

    # Handle clear button
    if clear_button:
        st.session_state.total_processed = 0
        st.session_state.total_detections = 0
        st.rerun()

    # Update config values
    cfg.CONFIDENCE_THRESHOLD = confidence_threshold
    cfg.OVERLAY_ALPHA = overlay_alpha

    # ==================== RESULTS SECTION ====================
    if uploaded_file is not None and detect_button:
        detector = load_model()
        results = process_image(detector, uploaded_file)

        if results is None:
            st.warning("⚠️ Image could not be processed. Please upload a valid satellite or aerial image.")
        else:
            # Update session stats
            st.session_state.total_processed += 1
            if results['metrics']['has_spill']:
                st.session_state.total_detections += 1

            # Add record to local session database
            add_detection_record(
                filename=uploaded_file.name,
                has_spill=results['metrics']['has_spill'],
                coverage_pct=results['metrics']['coverage_percentage'],
                avg_confidence=results['metrics']['avg_confidence'],
                max_confidence=results['metrics']['max_confidence'],
                detected_pixels=results['metrics']['detected_pixels']
            )

            # Ensure all images are uint8 for proper color display
            original_img = ensure_uint8(results['original_image'])
            binary_mask = ensure_uint8(results['binary_mask'])
            
            # Create visualizations with proper color handling
            overlay = create_overlay(
                original_img,
                binary_mask,
                alpha=overlay_alpha
            )
            overlay = ensure_uint8(overlay)
            
            heatmap = create_confidence_heatmap(results['confidence_map'], original_img)
            heatmap = ensure_uint8(heatmap)

            # Convert to PIL Images for storage upload
            overlay_pil = Image.fromarray(overlay)
            heatmap_pil = Image.fromarray(heatmap)
            binary_mask_pil = Image.fromarray(binary_mask)

            # Save to Supabase database with images
            save_success = False
            if overlay is not None and heatmap is not None and binary_mask is not None:
                save_success = save_to_supabase_with_images(
                    filename=uploaded_file.name,
                    has_spill=bool(results['metrics']['has_spill']),
                    coverage_pct=float(results['metrics']['coverage_percentage']),
                    avg_confidence=float(results['metrics']['avg_confidence']),
                    max_confidence=float(results['metrics']['max_confidence']),
                    detected_pixels=int(results['metrics']['detected_pixels']),
                    overlay_img=overlay_pil,
                    heatmap_img=heatmap_pil,
                    binary_mask_img=binary_mask_pil
                )
                
                if save_success:
                    st.success("✅ Detection data and images saved to cloud database!")
                else:
                    st.warning("⚠️ Images processed but cloud upload had issues. Check logs.")

            # *** ENHANCED DETECTION STATUS - REPLACES OLD BINARY DISPLAY ***
            display_enhanced_detection_status(results, uploaded_file.name)

            # Results in properly aligned columns
            col1, col2, col3 = st.columns(3, gap="medium")
            
            # Column 1: Detection Overlay
            with col1:
                st.markdown('<h3 style="font-family: Poppins, sans-serif; color: #ffffff; margin-bottom: 20px; font-size: 1.5rem; font-weight: 900; text-align: center; padding: 16px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-radius: 12px; box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);">Detection Overlay</h3>', unsafe_allow_html=True)
                st.image(overlay, use_column_width=True, channels="RGB")
                st.download_button(
                    "📥 Download Overlay",
                    data=image_to_bytes(overlay_pil),
                    file_name='oil_spill_overlay.png',
                    mime='image/png',
                    use_container_width=True
                )

            # Column 2: Confidence Heatmap
            with col2:
                st.markdown('<h3 style="font-family: Poppins, sans-serif; color: #ffffff; margin-bottom: 20px; font-size: 1.5rem; font-weight: 900; text-align: center; padding: 16px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-radius: 12px; box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);">Confidence Heatmap</h3>', unsafe_allow_html=True)
                st.image(heatmap, use_column_width=True, channels="RGB")
                st.download_button(
                    "📥 Download Heatmap",
                    data=image_to_bytes(heatmap_pil),
                    file_name='oil_spill_heatmap.png',
                    mime='image/png',
                    use_container_width=True
                )

            # Column 3: Metrics
            with col3:
                st.markdown('<h3 style="font-family: Poppins, sans-serif; color: #ffffff; margin-bottom: 20px; font-size: 1.5rem; font-weight: 900; text-align: center; padding: 16px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-radius: 12px; box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);">Detection Metrics</h3>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['coverage_percentage']:.2f}%</div>
                    <div class="metric-label">Spill Coverage Area</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['avg_confidence']:.1%}</div>
                    <div class="metric-label">Average Confidence Level</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['max_confidence']:.1%}</div>
                    <div class="metric-label">Maximum Confidence Level</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['detected_pixels']:,}</div>
                    <div class="metric-label">Total Detected Pixels</div>
                </div>
                """, unsafe_allow_html=True)

            # Additional details in tabs
            st.markdown("<br>", unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(["📊 Binary Mask", "📋 Raw JSON Data", "📈 Analysis Summary"])

            with tab1:
                st.markdown('<div class="tab-content-box">', unsafe_allow_html=True)
                st.image(
                    binary_mask,
                    caption='Binary segmentation mask (white = oil spill detected)',
                    use_column_width=True,
                    clamp=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.markdown('<div class="tab-content-box">', unsafe_allow_html=True)
                st.json({
                    'detection_status': 'Spill Detected' if results['metrics']['has_spill'] else 'No Spill',
                    'coverage_percentage': float(results['metrics']['coverage_percentage']),
                    'detected_pixels': int(results['metrics']['detected_pixels']),
                    'total_pixels': int(results['metrics']['total_pixels']),
                    'average_confidence': float(results['metrics']['avg_confidence']),
                    'maximum_confidence': float(results['metrics']['max_confidence']),
                    'threshold_used': float(confidence_threshold),
                    'overlay_alpha': float(overlay_alpha)
                })
                st.markdown('</div>', unsafe_allow_html=True)

            with tab3:
                st.markdown('<div class="tab-content-box"><div class="analysis-summary">', unsafe_allow_html=True)
                
                st.markdown(f"""
                ### Analysis Summary
                
                **Detection Result:** {'✅ Oil Spill Detected' if results['metrics']['has_spill'] else '❌ No Oil Spill Detected'}
                
                **Coverage Analysis:**
                - Total area analyzed: <strong>{results['metrics']['total_pixels']:,} pixels</strong>
                - Contaminated area: <strong>{results['metrics']['detected_pixels']:,} pixels</strong>
                - Coverage percentage: <strong>{results['metrics']['coverage_percentage']:.2f}%</strong>
                
                **Confidence Metrics:**
                - Average confidence: <strong>{results['metrics']['avg_confidence']:.1%}</strong>
                - Maximum confidence: <strong>{results['metrics']['max_confidence']:.1%}</strong>
                - Detection threshold: <strong>{confidence_threshold:.1%}</strong>
                
                **Recommendations:**
                {('<strong>- Immediate response required for cleanup operations</strong>' if results['metrics']['coverage_percentage'] > 5 else '<strong>- Monitor the area for potential expansion</strong>') if results['metrics']['has_spill'] else '<strong>- Continue routine monitoring</strong>'}
                """, unsafe_allow_html=True)
                
                st.markdown('</div></div>', unsafe_allow_html=True)

    # ==================== STATISTICS BANNER ====================
    st.markdown(f"""
    <div class="stats-banner">
        <div class="stat">
            <div class="stat-number">{st.session_state.total_processed}</div>
            <div class="stat-label">Images Analyzed</div>
        </div>
        <div class="stat">
            <div class="stat-number">{st.session_state.total_detections}</div>
            <div class="stat-label">Spills Detected</div>
        </div>
        <div class="stat">
            <div class="stat-number">{(st.session_state.total_detections / max(st.session_state.total_processed, 1) * 100):.1f}%</div>
            <div class="stat-label">Detection Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== LIVE DATABASE SECTION ====================
    if st.session_state.detection_records:
        
        st.markdown("""
        <div class="database-header">
            <h2 class="section-title">📊 Live Detection Database (Session)</h2>
        </div>
        """, unsafe_allow_html=True)
        
        df = get_records_dataframe()
        total_records = len(df)
        spills_found = len(df[df['result'].str.contains('✅')])
        avg_coverage = df['coverage_%'].mean() if not df.empty else 0
        
        st.markdown(f"""
        <div class="db-stats">
            <div class="db-stat-box">
                <div class="db-stat-value">{total_records}</div>
                <div class="db-stat-label">Total Records</div>
            </div>
            <div class="db-stat-box">
                <div class="db-stat-value">{spills_found}</div>
                <div class="db-stat-label">Spills Found</div>
            </div>
            <div class="db-stat-box">
                <div class="db-stat-value">{avg_coverage:.1f}%</div>
                <div class="db-stat-label">Avg Coverage</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "filename": st.column_config.TextColumn("Image File", width="medium"),
                "result": st.column_config.TextColumn("Result", width="small"),
                "coverage_%": st.column_config.NumberColumn("Coverage %", format="%.2f"),
                "avg_confidence": st.column_config.NumberColumn("Avg Conf %", format="%.1f"),
                "max_confidence": st.column_config.NumberColumn("Max Conf %", format="%.1f"),
                "detected_pixels": st.column_config.NumberColumn("Pixels", format="%d")
            }
        )
        
        col_export1, col_export2, col_export3 = st.columns([1, 1, 2])
        with col_export1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export CSV",
                data=csv,
                file_name=f'detection_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                use_container_width=True
            )
        with col_export2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Export JSON",
                data=json_data,
                file_name=f'detection_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json',
                use_container_width=True
            )
        with col_export3:
            if st.button("🗑️ Clear All Records", use_container_width=True):
                st.session_state.detection_records = []
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="database-section">
            <h2 class="section-title">📊 Live Detection Database (Session)</h2>
            <p style="text-align: center; color: #4A5568; padding: 40px 0; font-size: 1.1rem;">
                No records yet. Upload and analyze images to see detection history here.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ==================== IMAGE GALLERY SECTION ====================
    st.markdown('<div style="margin-top: 60px;"></div>', unsafe_allow_html=True)
    display_detection_image_gallery()

    # ==================== PREVIOUS DETECTIONS FROM SUPABASE ====================
    st.markdown('<div style="margin-top: 60px;"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="database-header">
        <h2 class="section-title">📊 Previous Detections (Database Records)</h2>
        <p style="color: #1E1E1E; font-size: 1.15rem; margin-top: 10px; font-weight: 600;">
            View detection metadata from the cloud database
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        if supabase is None:
            st.info("⚠️ Supabase database not configured. Only session data is available.")
        else:
            # Add "View All Detections" button
            col_view_option1, col_view_option2 = st.columns([1, 3])
            
            with col_view_option1:
                view_all_button = st.button("📋 View All Detections", use_container_width=True, type="primary")
            
            with col_view_option2:
                st.markdown('<p style="color: #4A5568; font-size: 0.95rem; margin-top: 10px;">Click to view all previous detections or select a date below to filter</p>', unsafe_allow_html=True)
            
            # Load all detections
            with st.spinner("Loading detections from database..."):
                all_data = fetch_all_detections("oil_detections")
            
            if not all_data or len(all_data) == 0:
                st.info("🔭 No detections found in the database. Upload and analyze images to populate the database.")
            else:
                # Extract unique dates
                dates = sorted(set([d.get('timestamp', '')[:10] for d in all_data if d.get('timestamp')]), reverse=True)
                
                if not dates:
                    st.warning("⚠️ No valid dates found in detection records.")
                else:
                    # Show data based on user choice
                    if view_all_button:
                        st.session_state['show_all_detections'] = True
                    
                    # Initialize session state for view all
                    if 'show_all_detections' not in st.session_state:
                        st.session_state['show_all_detections'] = False
                    
                    if st.session_state['show_all_detections']:
                        # Display ALL detections
                        st.markdown('<div style="background: #e8f5e9; padding: 15px; border-radius: 10px; margin: 20px 0;"><p style="color: #2e7d32; font-weight: 600; margin: 0;">📋 Viewing All Detections</p></div>', unsafe_allow_html=True)
                        
                        df_supabase = pd.DataFrame(all_data)
                        
                        if 'timestamp' in df_supabase.columns:
                            df_supabase['timestamp'] = pd.to_datetime(df_supabase['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        if 'has_spill' in df_supabase.columns:
                            df_supabase['result'] = df_supabase['has_spill'].apply(lambda x: 'Spill Detected ✅' if x else 'No Spill ❌')
                        
                        total_db_records = len(df_supabase)
                        spills_in_db = df_supabase['has_spill'].sum() if 'has_spill' in df_supabase.columns else 0
                        avg_coverage_db = df_supabase['coverage_percentage'].mean() if 'coverage_percentage' in df_supabase.columns else 0
                        
                        st.markdown(f"""
                        <div class="db-stats">
                            <div class="db-stat-box">
                                <div class="db-stat-value">{total_db_records}</div>
                                <div class="db-stat-label">Total Detections</div>
                            </div>
                            <div class="db-stat-box">
                                <div class="db-stat-value">{spills_in_db}</div>
                                <div class="db-stat-label">Spills Detected</div>
                            </div>
                            <div class="db-stat-box">
                                <div class="db-stat-value">{avg_coverage_db:.1f}%</div>
                                <div class="db-stat-label">Avg Coverage</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show dataframe
                        display_columns = []
                        column_config = {}
                        
                        if 'timestamp' in df_supabase.columns:
                            display_columns.append('timestamp')
                            column_config['timestamp'] = st.column_config.TextColumn("Timestamp", width="medium")
                        
                        if 'filename' in df_supabase.columns:
                            display_columns.append('filename')
                            column_config['filename'] = st.column_config.TextColumn("Image File", width="medium")
                        
                        if 'result' in df_supabase.columns:
                            display_columns.append('result')
                            column_config['result'] = st.column_config.TextColumn("Result", width="small")
                        
                        if 'coverage_percentage' in df_supabase.columns:
                            display_columns.append('coverage_percentage')
                            column_config['coverage_percentage'] = st.column_config.NumberColumn("Coverage %", format="%.2f")
                        
                        if 'avg_confidence' in df_supabase.columns:
                            display_columns.append('avg_confidence')
                            column_config['avg_confidence'] = st.column_config.NumberColumn("Avg Confidence", format="%.3f")
                        
                        if 'max_confidence' in df_supabase.columns:
                            display_columns.append('max_confidence')
                            column_config['max_confidence'] = st.column_config.NumberColumn("Max Confidence", format="%.3f")
                        
                        if 'detected_pixels' in df_supabase.columns:
                            display_columns.append('detected_pixels')
                            column_config['detected_pixels'] = st.column_config.NumberColumn("Detected Pixels", format="%d")
                        
                        st.dataframe(
                            df_supabase[display_columns] if display_columns else df_supabase,
                            use_container_width=True,
                            hide_index=True,
                            column_config=column_config
                        )
                        
                        col_db1, col_db2, col_db3 = st.columns(3)
                        with col_db1:
                            csv_db = df_supabase.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Export All (CSV)",
                                data=csv_db,
                                file_name=f'all_detections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
                        with col_db2:
                            json_db = df_supabase.to_json(orient='records', indent=2)
                            st.download_button(
                                "📥 Export All (JSON)",
                                data=json_db,
                                file_name=f'all_detections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                                mime='application/json',
                                use_container_width=True
                            )
                        with col_db3:
                            if st.button("🔄 Back to Date Filter", use_container_width=True):
                                st.session_state['show_all_detections'] = False
                                st.rerun()
                    
                    else:
                        # DATE FILTER MODE
                        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                        st.markdown('<h3>📅 Select Date to View Detections</h3>', unsafe_allow_html=True)
                        
                        selected_date = st.selectbox(
                            "Choose a date:",
                            options=["-- Select a Date --"] + dates,
                            index=0,
                            key="prev_detection_date_filter"
                        )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        if selected_date == "-- Select a Date --":
                            st.info("👆 Please select a date above to view detection records, or click 'View All Detections' button.")
                        else:
                            # Filter data by selected date
                            filtered_data = [d for d in all_data if d.get('timestamp', '').startswith(selected_date)]
                            
                            if not filtered_data:
                                st.warning(f"No detections found for date: {selected_date}")
                            else:
                                df_supabase = pd.DataFrame(filtered_data)
                                
                                if 'timestamp' in df_supabase.columns:
                                    df_supabase['timestamp'] = pd.to_datetime(df_supabase['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                                
                                if 'has_spill' in df_supabase.columns:
                                    df_supabase['result'] = df_supabase['has_spill'].apply(lambda x: 'Spill Detected ✅' if x else 'No Spill ❌')
                                
                                total_db_records = len(df_supabase)
                                spills_in_db = df_supabase['has_spill'].sum() if 'has_spill' in df_supabase.columns else 0
                                avg_coverage_db = df_supabase['coverage_percentage'].mean() if 'coverage_percentage' in df_supabase.columns else 0
                                
                                st.markdown(f"""
                                <div class="db-stats">
                                    <div class="db-stat-box">
                                        <div class="db-stat-value">{total_db_records}</div>
                                        <div class="db-stat-label">Detections on {selected_date}</div>
                                    </div>
                                    <div class="db-stat-box">
                                        <div class="db-stat-value">{spills_in_db}</div>
                                        <div class="db-stat-label">Spills Detected</div>
                                    </div>
                                    <div class="db-stat-box">
                                        <div class="db-stat-value">{avg_coverage_db:.1f}%</div>
                                        <div class="db-stat-label">Avg Coverage</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                display_columns = []
                                column_config = {}
                                
                                if 'timestamp' in df_supabase.columns:
                                    display_columns.append('timestamp')
                                    column_config['timestamp'] = st.column_config.TextColumn("Timestamp", width="medium")
                                
                                if 'filename' in df_supabase.columns:
                                    display_columns.append('filename')
                                    column_config['filename'] = st.column_config.TextColumn("Image File", width="medium")
                                
                                if 'result' in df_supabase.columns:
                                    display_columns.append('result')
                                    column_config['result'] = st.column_config.TextColumn("Result", width="small")
                                
                                if 'coverage_percentage' in df_supabase.columns:
                                    display_columns.append('coverage_percentage')
                                    column_config['coverage_percentage'] = st.column_config.NumberColumn("Coverage %", format="%.2f")
                                
                                if 'avg_confidence' in df_supabase.columns:
                                    display_columns.append('avg_confidence')
                                    column_config['avg_confidence'] = st.column_config.NumberColumn("Avg Confidence", format="%.3f")
                                
                                if 'max_confidence' in df_supabase.columns:
                                    display_columns.append('max_confidence')
                                    column_config['max_confidence'] = st.column_config.NumberColumn("Max Confidence", format="%.3f")
                                
                                if 'detected_pixels' in df_supabase.columns:
                                    display_columns.append('detected_pixels')
                                    column_config['detected_pixels'] = st.column_config.NumberColumn("Detected Pixels", format="%d")
                                
                                st.dataframe(
                                    df_supabase[display_columns] if display_columns else df_supabase,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config=column_config
                                )
                                
                                col_db1, col_db2 = st.columns(2)
                                with col_db1:
                                    csv_db = df_supabase.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        f"📥 Export {selected_date} (CSV)",
                                        data=csv_db,
                                        file_name=f'detections_{selected_date}.csv',
                                        mime='text/csv',
                                        use_container_width=True
                                    )
                                with col_db2:
                                    json_db = df_supabase.to_json(orient='records', indent=2)
                                    st.download_button(
                                        f"📥 Export {selected_date} (JSON)",
                                        data=json_db,
                                        file_name=f'detections_{selected_date}.json',
                                        mime='application/json',
                                        use_container_width=True
                                    )
    
    except Exception as e:
        st.error(f"❌ Error loading previous detections: {str(e)}")
        print(f"Database error: {e}")

    # ==================== FOOTER ====================
    st.markdown("""
    <div class="footer">
        <p style="margin-bottom: 10px;">🛰️ <strong>AquaScan Intelligence</strong> - Safeguarding Marine Environments Through AI</p>
        <p class="author"><a href="https://www.linkedin.com/in/ayush-nema-jec/" target="_blank" style="color: inherit; text-decoration: none;">Built with precision by Ayush Nema</a></p>    
        <p style="margin-top: 10px; font-size: 0.95rem; opacity: 0.9;">
             🎨 Streamlit • ☁️ Supabase • 🧠 U-Net CNN
        </p>
        <p style="margin-top: 15px; font-size: 0.85rem; opacity: 0.8;">
            © 2026 AquaScan Intelligence Platform. Licensed for environmental research and marine protection initiatives.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()