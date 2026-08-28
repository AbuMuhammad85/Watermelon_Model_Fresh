import os
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import convolve

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ImageQualityGate:
    def __init__(self, blur_threshold=60.0, dark_threshold=40.0, bright_threshold=235.0, min_green_ratio=0.20):
        """
        Initializes the quality gate with configurable thresholds.
        
        Args:
            blur_threshold (float): Laplacian variance below this value is considered blurry.
            dark_threshold (float): Average grayscale value below this is considered too dark.
            bright_threshold (float): Average grayscale value above this is considered too bright.
            min_green_ratio (float): Minimum ratio of green pixels required to assume a leaf is present.
        """
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.min_green_ratio = min_green_ratio

    def analyze_image(self, pil_image):
        """
        Analyzes the image for quality checks.
        
        Args:
            pil_image (PIL.Image): The input image to inspect.
            
        Returns:
            dict: Diagnostic metrics and boolean flags.
        """
        # Convert to RGB to ensure standard 3-channel structure
        img_rgb = pil_image.convert("RGB")
        img_np = np.array(img_rgb).astype(np.float32)
        
        # 1. Grayscale conversion for blur and exposure checks
        img_gray_pil = img_rgb.convert("L")
        img_gray = np.array(img_gray_pil).astype(np.float32)
        
        # 2. Exposure check (Darkness / Brightness)
        mean_brightness = float(np.mean(img_gray))
        is_dark = mean_brightness < self.dark_threshold
        is_bright = mean_brightness > self.bright_threshold
        
        # 3. Blur Check (Laplacian Variance)
        # 3x3 Laplacian kernel
        laplacian_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        
        laplacian = convolve(img_gray, laplacian_kernel, mode="reflect")
        blur_variance = float(np.var(laplacian))
        is_blurry = blur_variance < self.blur_threshold
        
        # 4. Framing Check / Visible Leaf Area
        # Excess Green Index (ExG): 2*G - R - B
        r = img_np[:, :, 0]
        g = img_np[:, :, 1]
        b = img_np[:, :, 2]
        exg = 2.0 * g - r - b
        
        # Threshold: pixel is green if ExG > 15.0 and green channel has some brightness (not dark shadow)
        green_mask = (exg > 15.0) & (g > 35.0)
        green_pixel_count = np.sum(green_mask)
        total_pixels = img_np.shape[0] * img_np.shape[1]
        green_ratio = float(green_pixel_count / total_pixels)
        
        is_poorly_framed = green_ratio < self.min_green_ratio
        
        # Gather reasons
        reasons = []
        if is_blurry:
            reasons.append("blurry")
        if is_dark:
            reasons.append("too_dark")
        if is_bright:
            reasons.append("too_bright")
        if is_poorly_framed:
            reasons.append("leaf_too_small")
            
        passed = len(reasons) == 0
        
        return {
            "passed": passed,
            "is_blurry": is_blurry,
            "blur_score": blur_variance,
            "is_dark": is_dark,
            "is_bright": is_bright,
            "brightness_score": mean_brightness,
            "is_poorly_framed": is_poorly_framed,
            "leaf_area_ratio": green_ratio,
            "reasons": reasons
        }

if __name__ == "__main__":
    # Test gate on a dummy image
    gate = ImageQualityGate()
    dummy_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    metrics = gate.analyze_image(dummy_img)
    print("Dummy Image Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
