# ============================================================================
# PREPROCESSING.PY — SAFE IMAGE PREPROCESSING (STREAMLIT + PYTORCH)
# ============================================================================

import cv2
import numpy as np
from PIL import Image
import io
import config.config as cfg


# ----------------------------------------------------------------------------
# Load & preprocess image (using PIL to avoid OpenCV issues)
# ----------------------------------------------------------------------------
def load_and_preprocess_image(image_input, target_size=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT)):
    """
    Convert image input → numpy → resize → normalize
    Supports: bytes, PIL Image, numpy array, file path
    """

    # 1️⃣ Convert input to PIL Image first
    if isinstance(image_input, bytes):
        pil_image = Image.open(io.BytesIO(image_input)).convert("RGB")

    elif isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB")

    elif isinstance(image_input, np.ndarray):
        pil_image = Image.fromarray(image_input).convert("RGB")

    elif isinstance(image_input, str):
        pil_image = Image.open(image_input).convert("RGB")

    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    # 2️⃣ Keep original as numpy array
    original_img = np.array(pil_image, dtype=np.uint8)
    
    # 3️⃣ Resize using PIL (safer than OpenCV)
    resized_pil = pil_image.resize(target_size, Image.Resampling.LANCZOS)
    resized = np.array(resized_pil, dtype=np.float32) / 255.0

    # 4️⃣ Add batch dimension → (1, H, W, C)
    resized = np.expand_dims(resized, axis=0)

    return resized, original_img


# ----------------------------------------------------------------------------
# Postprocess prediction mask (using PIL for resize)
# ----------------------------------------------------------------------------
def postprocess_mask(pred_mask, threshold=cfg.CONFIDENCE_THRESHOLD, target_size=None):
    """
    Convert model output → binary mask + confidence map
    """
    
    # Ensure it's a numpy array
    if not isinstance(pred_mask, np.ndarray):
        pred_mask = np.array(pred_mask)
    
    # Remove batch/channel dims if present
    while len(pred_mask.shape) > 2:
        pred_mask = pred_mask.squeeze()
    
    # Ensure 2D and float32
    if pred_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {pred_mask.shape}")
    
    pred_mask = pred_mask.astype(np.float32)
    
    # Create confidence map copy
    confidence_map = pred_mask.copy()
    
    # Binary mask (0 or 255)
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
    
    # Resize back to original image size if needed (using PIL)
    if target_size is not None:
        # target_size is (width, height)
        w, h = int(target_size[0]), int(target_size[1])
        
        # Resize binary mask using PIL
        binary_pil = Image.fromarray(binary_mask, mode='L')
        binary_pil = binary_pil.resize((w, h), Image.Resampling.NEAREST)
        binary_mask = np.array(binary_pil, dtype=np.uint8)
        
        # Resize confidence map using PIL
        # Normalize to 0-255 for PIL, then back to 0-1
        conf_uint8 = (confidence_map * 255).astype(np.uint8)
        conf_pil = Image.fromarray(conf_uint8, mode='L')
        conf_pil = conf_pil.resize((w, h), Image.Resampling.BILINEAR)
        confidence_map = np.array(conf_pil, dtype=np.float32) / 255.0
    
    return binary_mask, confidence_map


# ----------------------------------------------------------------------------
# Validate uploaded image
# ----------------------------------------------------------------------------
def validate_image(image_bytes, max_size_mb=10):
    try:
        if not image_bytes:
            return False, "Uploaded file is empty"

        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File too large ({size_mb:.2f} MB)"

        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Validate integrity

        return True, "Valid image"

    except Exception as e:
        return False, f"Invalid image: {str(e)}"
