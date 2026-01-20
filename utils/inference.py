# =============================================================================
# utils/inference.py — OpenCV-safe, model-aligned inference
# =============================================================================

import torch
import numpy as np
import os
import requests
from pathlib import Path

from models.model_architecture import EnhancedUNet
from utils.preprocessing import load_and_preprocess_image, postprocess_mask

# Hugging Face model URL
HUGGINGFACE_MODEL_URL = "https://huggingface.co/a-nema/oil-spill-model/resolve/main/best_model.pth"


def download_model(model_path, url=HUGGINGFACE_MODEL_URL):
    """Download model from Hugging Face if not exists"""
    model_path = Path(model_path)
    
    if model_path.exists():
        return str(model_path)
    
    print(f"📥 Downloading model from Hugging Face...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(model_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\r📥 Downloading: {pct:.1f}%", end="", flush=True)
    
    print(f"\n✅ Model downloaded to {model_path}")
    return str(model_path)


class OilSpillDetector:
    def __init__(self, model_path="./models/best_model.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Download model if not exists (for Streamlit Cloud)
        model_path = download_model(model_path)

        # Load model
        self.model = EnhancedUNet(in_channels=3, out_channels=1)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # IMPORTANT: load ONLY state_dict
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_input):
        """
        Run inference on one image (bytes / PIL / numpy)
        """

        try:
            # ----------------------------
            # 1. Preprocess image
            # ----------------------------
            preprocessed, original = load_and_preprocess_image(image_input)

            # Convert to torch tensor (N, C, H, W)
            input_tensor = torch.from_numpy(preprocessed).permute(0, 3, 1, 2)
            input_tensor = input_tensor.to(self.device)

            # ----------------------------
            # 2. Model inference
            # ----------------------------
            with torch.no_grad():
                output = self.model(input_tensor)

            # ----------------------------
            # 3. Convert output to numpy
            # ----------------------------
            pred_mask = output.squeeze().cpu().numpy().astype(np.float32)

            if pred_mask.ndim != 2:
                raise ValueError(f"Expected 2D mask, got shape {pred_mask.shape}")

            # ----------------------------
            # 4. Postprocess
            # ----------------------------
            # Ensure original is a valid numpy array
            if not isinstance(original, np.ndarray):
                original = np.array(original, dtype=np.uint8)
            original = np.ascontiguousarray(original)
            
            h, w = original.shape[:2]

            binary_mask, confidence_map = postprocess_mask(
                pred_mask,
                threshold=0.5,
                target_size=(w, h)  # (width, height)
            )

            metrics = self._calculate_metrics(binary_mask, confidence_map)

            return {
                "binary_mask": binary_mask,
                "confidence_map": confidence_map,
                "original_image": original,
                "metrics": metrics
            }

        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")

    def _calculate_metrics(self, binary_mask, confidence_map):
        """Calculate detection metrics"""
        total_pixels = binary_mask.size
        detected_pixels = int(np.sum(binary_mask > 0))
        
        coverage_percentage = (detected_pixels / total_pixels) * 100
        
        if detected_pixels > 0:
            avg_confidence = float(np.mean(confidence_map[binary_mask > 0]))
        else:
            avg_confidence = 0.0
        
        max_confidence = float(np.max(confidence_map))
        
        return {
            'coverage_percentage': float(coverage_percentage),
            'detected_pixels': detected_pixels,
            'total_pixels': int(total_pixels),
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'has_spill': detected_pixels > 0
        }


# =============================================================================
# SINGLETON
# =============================================================================

_detector_instance = None

def get_detector(model_path="./models/enhanced_unet_architecture.pth"):
    """Get or create detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = OilSpillDetector(model_path)
    return _detector_instance
