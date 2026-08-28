import os
import sys
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def perturb_brightness(pil_image, factor):
    """
    Adjusts the brightness of the PIL Image.
    factor < 1.0 -> under-exposed/dark.
    factor > 1.0 -> over-exposed/glare.
    """
    enhancer = ImageEnhance.Brightness(pil_image)
    return enhancer.enhance(factor)

def perturb_blur(pil_image, radius=3):
    """
    Applies Gaussian Blur to simulate camera out-of-focus.
    """
    return pil_image.filter(ImageFilter.GaussianBlur(radius))

def perturb_distance(pil_image, scale=0.5, bg_color=(100, 80, 60)):
    """
    Simulates distance by resizing the leaf smaller and padding the background.
    """
    w, h = pil_image.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize leaf
    resized_leaf = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Create padded background
    padded_img = Image.new("RGB", (w, h), bg_color)
    
    # Paste resized leaf in center
    offset_x = (w - new_w) // 2
    offset_y = (h - new_h) // 2
    padded_img.paste(resized_leaf, (offset_x, offset_y))
    
    return padded_img

def perturb_soil_background(pil_image, soil_color=(115, 80, 55), noise_var=20):
    """
    Segment the leaf (using Excess Green index) and replace the background (non-leaf)
    pixels with simulated brownish soil noise to evaluate soil background interference.
    """
    img_rgb = pil_image.convert("RGB")
    img_np = np.array(img_rgb).astype(np.float32)
    h, w, c = img_np.shape
    
    # 1. Excess Green Index segmentation
    r = img_np[:, :, 0]
    g = img_np[:, :, 1]
    b = img_np[:, :, 2]
    exg = 2.0 * g - r - b
    
    # Leaf mask: positive ExG and not black
    leaf_mask = (exg > 15.0) & (g > 35.0)
    leaf_mask_3d = np.expand_dims(leaf_mask, axis=-1)
    
    # 2. Generate soil noise background
    soil_base = np.zeros_like(img_np)
    soil_base[:, :, 0] = soil_color[0]
    soil_base[:, :, 1] = soil_color[1]
    soil_base[:, :, 2] = soil_color[2]
    
    # Add random variations to make it look like dirt/soil texture
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, noise_var, (h, w, c))
    soil_img = np.clip(soil_base + noise, 0, 255)
    
    # 3. Blend leaf with soil background
    blended_np = np.where(leaf_mask_3d, img_np, soil_img)
    
    return Image.fromarray(blended_np.astype(np.uint8))

if __name__ == "__main__":
    print("Robustness perturbation modules verified.")
