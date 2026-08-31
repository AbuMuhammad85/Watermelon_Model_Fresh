import os
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_fill_holes, sobel, convolve
from scipy.spatial import ConvexHull

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LeafAuthenticityGate:
    """
    Offline heuristic safety filter that evaluates whether an input image visually
    resembles a watermelon leaf prior to running disease classification.
    
    NOTE:
    This is an offline heuristic computer-vision filter (geometry, color indices,
    contour complexity, and texture gradients), NOT a trained machine learning classifier
    or scientific model. It is designed to conservatively filter out obvious non-leaf
    objects and whole watermelon fruits to prevent false confident disease predictions.
    """
    
    def __init__(self,
                 min_vegetation_ratio=0.03,
                 fruit_min_solidity=0.88,
                 fruit_min_extent=0.70,
                 fruit_max_edge_density=0.18,
                 min_leaf_score=0.35):
        """
        Initializes the Leaf Authenticity Gate with conservative heuristic thresholds.
        
        Args:
            min_vegetation_ratio (float): Minimum proportion of image area showing plant/vegetation tissue.
                Set conservatively (0.03 = 3%) to avoid rejecting distant or severely necrotic leaves.
            fruit_min_solidity (float): Solidity threshold (area / convex hull) characteristic of solid
                convex shapes like watermelon fruits (leaves have open sinuses/lobes, lower solidity).
            fruit_min_extent (float): Extent threshold (area / bounding box) characteristic of solid
                ellipsoids (leaves have open margins, lower extent).
            fruit_max_edge_density (float): Edge density threshold below which an object lacks leaf
                vein or lesion complexity (fruit rind is relatively smooth and waxy).
            min_leaf_score (float): Composite multi-signal threshold required to pass the leaf authenticity gate.
        """
        self.min_vegetation_ratio = min_vegetation_ratio
        self.fruit_min_solidity = fruit_min_solidity
        self.fruit_min_extent = fruit_min_extent
        self.fruit_max_edge_density = fruit_max_edge_density
        self.min_leaf_score = min_leaf_score

    def analyze_leaf(self, pil_image):
        """
        Analyzes the given image to check if it exhibits watermelon leaf characteristics
        or if it resembles a whole watermelon fruit or non-leaf object.
        
        Args:
            pil_image (PIL.Image): Input image.
            
        Returns:
            dict: Structured result containing:
                - is_leaf (bool): True if image passes authenticity check, False if rejected.
                - score (float): Composite leaf authenticity score [0.0, 1.0].
                - reason (str): Rejection reason code or 'passed'.
                - message (str): Clear user-facing message in English.
                - message_hausa (str): Clear user-facing message in Hausa.
                - metrics (dict): Breakdown of computed geometric and texture metrics.
        """
        # Standardize working size (512x512) for consistent scale-invariant metric calculation
        img_512 = pil_image.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR)
        arr = np.array(img_512, dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        
        # ---------------------------------------------------------------------
        # 1. Plant & Vegetation Tissue Segmentation (Excess Green & Necrosis)
        # ---------------------------------------------------------------------
        # Excess Green Index (ExG): 2*G - R - B
        exg = 2.0 * g - r - b
        
        # Green healthy/mildly chlorotic vegetation tissue
        green_tissue = (exg > 8.0) & (g > 30.0)
        
        # Chlorotic/yellow leaf tissue (high G and R, low B)
        yellow_tissue = (g > b * 1.15) & (r > b * 1.10) & (g > 35.0) & (r > 35.0) & (exg > -25.0)
        
        # Necrotic/brown lesion tissue attached to watermelon leaf blades (e.g. Anthracnose / Downy Mildew spots)
        necrotic_tissue = (r > b * 1.05) & (g > b * 1.05) & (r > 25.0) & (g > 25.0) & (exg > -35.0)
        
        # Combined plant tissue mask
        plant_mask = green_tissue | yellow_tissue | necrotic_tissue
        plant_ratio = float(np.mean(plant_mask))
        green_ratio = float(np.mean(green_tissue))
        
        # Immediate rejection if almost no plant/vegetation tissue is present in the frame
        if plant_ratio < self.min_vegetation_ratio:
            return {
                "is_leaf": False,
                "score": 0.0,
                "reason": "insufficient_vegetation",
                "message": "Please capture a clear watermelon leaf, not the watermelon fruit.",
                "message_hausa": "Da fatan za a dauki hoton ganyen kankana mai kyau, ba 'ya'yan kankana ba.",
                "metrics": {
                    "plant_ratio": plant_ratio,
                    "green_ratio": green_ratio,
                    "comp_area_ratio": 0.0,
                    "solidity": 0.0,
                    "extent": 0.0,
                    "aspect_ratio": 1.0,
                    "boundary_complexity": 0.0,
                    "internal_edge_density": 0.0,
                    "fruit_resemblance": 0.0,
                    "leaf_authenticity_score": 0.0
                }
            }
            
        # ---------------------------------------------------------------------
        # 2. Connected Component & Region Analysis
        # ---------------------------------------------------------------------
        # Prioritize green tissue for main contour extraction, fallback to plant mask
        primary_mask = green_tissue if np.sum(green_tissue) > 100 else plant_mask
        labeled, num_features = label(primary_mask)
        if num_features == 0:
            labeled, num_features = label(plant_mask)
            
        if num_features == 0:
            return {
                "is_leaf": False,
                "score": 0.0,
                "reason": "non_leaf_detected",
                "message": "Please capture a clear watermelon leaf, not the watermelon fruit.",
                "message_hausa": "Da fatan za a dauki hoton ganyen kankana mai kyau, ba 'ya'yan kankana ba.",
                "metrics": {
                    "plant_ratio": plant_ratio,
                    "green_ratio": green_ratio,
                    "comp_area_ratio": 0.0,
                    "solidity": 0.0,
                    "extent": 0.0,
                    "aspect_ratio": 1.0,
                    "boundary_complexity": 0.0,
                    "internal_edge_density": 0.0,
                    "fruit_resemblance": 0.0,
                    "leaf_authenticity_score": 0.0
                }
            }
            
        sizes = [int(np.sum(labeled == i)) for i in range(1, num_features + 1)]
        max_idx = int(np.argmax(sizes)) + 1
        main_comp = (labeled == max_idx)
        filled_comp = binary_fill_holes(main_comp)
        
        comp_area = float(np.sum(main_comp))
        filled_area = float(np.sum(filled_comp))
        comp_area_ratio = comp_area / (512.0 * 512.0)
        
        # ---------------------------------------------------------------------
        # 3. Shape, Convexity, & Contour Analysis
        # ---------------------------------------------------------------------
        pts = np.argwhere(main_comp)
        if len(pts) < 20:
            hull_area = comp_area
        else:
            try:
                hull = ConvexHull(pts)
                hull_area = float(hull.volume)  # In 2D, ConvexHull volume is the polygon area
            except Exception:
                hull_area = comp_area
                
        # Solidity = filled component area / convex hull area
        # A whole watermelon fruit is solid convex (solidity ~ 0.90 - 0.99)
        # A lobed leaf has deep sinuses between lobes (solidity usually lower, ~ 0.40 - 0.85)
        solidity = float(filled_area / hull_area) if hull_area > 0 else 1.0
        
        # Bounding box Extent & Aspect Ratio
        y_idx, x_idx = np.where(main_comp)
        h_box = float(np.max(y_idx) - np.min(y_idx) + 1)
        w_box = float(np.max(x_idx) - np.min(x_idx) + 1)
        box_area = h_box * w_box
        extent = float(comp_area / box_area) if box_area > 0 else 0.0
        aspect_ratio = float(max(w_box / h_box, h_box / w_box)) if min(h_box, w_box) > 0 else 1.0
        
        # Boundary complexity: ratio of actual perimeter to perimeter of a smooth circle of equivalent area
        # Leaves have high perimeter complexity due to lobes and serrations; smooth fruits have lower ratios
        sy = sobel(main_comp.astype(float), axis=0)
        sx = sobel(main_comp.astype(float), axis=1)
        perim = float(np.sum(np.sqrt(sx**2 + sy**2) > 0))
        min_perim = 2.0 * np.sqrt(np.pi * comp_area)
        boundary_complexity = float(perim / min_perim) if min_perim > 0 else 1.0
        
        # ---------------------------------------------------------------------
        # 4. Texture & Internal Edge Distribution
        # ---------------------------------------------------------------------
        # Watermelon leaves contain reticulate venation and lesion edges across the leaf body.
        # Smooth fruit rinds have low internal high-frequency edge density across their large surface.
        gray = np.array(img_512.convert('L'), dtype=np.float32)
        lap = np.abs(convolve(gray, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)))
        internal_edge_density = float(np.mean(lap[main_comp] > 8.0)) if comp_area > 0 else 0.0
        
        # ---------------------------------------------------------------------
        # 5. Multi-Signal Scoring & Fruit Resemblance
        # ---------------------------------------------------------------------
        # Sub-scores (each normalized to [0.0, 1.0])
        veg_score = min(1.0, max(0.0, (plant_ratio - 0.03) / 0.15))
        texture_score = min(1.0, max(0.0, (internal_edge_density - 0.08) / 0.15))
        shape_complexity_score = min(1.0, max(0.0, (boundary_complexity - 1.8) / 3.0))
        
        # Fruit Resemblance Heuristic:
        # A whole watermelon fruit is a monolithic solid object with:
        # - Substantial area (comp_area_ratio >= 0.10)
        # - High solidity (solidity >= fruit_min_solidity)
        # - High extent (extent >= fruit_min_extent)
        # - Low internal edge density (< fruit_max_edge_density)
        is_monolithic_solid = (solidity >= self.fruit_min_solidity) and (extent >= self.fruit_min_extent) and (comp_area_ratio >= 0.10)
        is_smooth_rind = (internal_edge_density < self.fruit_max_edge_density)
        
        fruit_resemblance = 0.0
        if is_monolithic_solid:
            solid_excess = (solidity - self.fruit_min_solidity) / (1.0 - self.fruit_min_solidity + 1e-5)
            extent_excess = (extent - self.fruit_min_extent) / (1.0 - self.fruit_min_extent + 1e-5)
            smooth_excess = max(0.0, (self.fruit_max_edge_density - internal_edge_density) / self.fruit_max_edge_density)
            fruit_resemblance = min(1.0, (solid_excess * 0.35 + extent_excess * 0.35 + smooth_excess * 0.30 + (0.4 if is_smooth_rind else 0.0)))
            
        # Composite Leaf Authenticity Score
        composite_leaf_score = (veg_score * 0.30) + (texture_score * 0.45) + (shape_complexity_score * 0.25)
        final_score = max(0.0, min(1.0, composite_leaf_score * (1.0 - fruit_resemblance * 0.90)))
        
        metrics = {
            "plant_ratio": plant_ratio,
            "green_ratio": green_ratio,
            "comp_area_ratio": comp_area_ratio,
            "solidity": solidity,
            "extent": extent,
            "aspect_ratio": aspect_ratio,
            "boundary_complexity": boundary_complexity,
            "internal_edge_density": internal_edge_density,
            "fruit_resemblance": fruit_resemblance,
            "leaf_authenticity_score": final_score
        }
        
        # ---------------------------------------------------------------------
        # 6. Safety Gate Decision Logic
        # ---------------------------------------------------------------------
        # Rule 1: High fruit resemblance -> reject specifically as fruit
        if is_monolithic_solid and is_smooth_rind and (fruit_resemblance > 0.50):
            return {
                "is_leaf": False,
                "score": final_score,
                "reason": "fruit_detected",
                "message": "Please capture a clear watermelon leaf, not the watermelon fruit.",
                "message_hausa": "Da fatan za a dauki hoton ganyen kankana mai kyau, ba 'ya'yan kankana ba.",
                "metrics": metrics
            }
            
        # Rule 2: Low composite score -> reject with appropriate reason code
        if final_score < self.min_leaf_score:
            reason = "fruit_detected" if fruit_resemblance > 0.40 else "insufficient_leaf_structure"
            return {
                "is_leaf": False,
                "score": final_score,
                "reason": reason,
                "message": "Please capture a clear watermelon leaf, not the watermelon fruit.",
                "message_hausa": "Da fatan za a dauki hoton ganyen kankana mai kyau, ba 'ya'yan kankana ba.",
                "metrics": metrics
            }
            
        # Rule 3: Passes leaf authenticity gate
        return {
            "is_leaf": True,
            "score": final_score,
            "reason": "passed",
            "message": "Leaf authenticity verified.",
            "message_hausa": "Ingancin ganyen kankana ya inganta.",
            "metrics": metrics
        }

if __name__ == "__main__":
    gate = LeafAuthenticityGate()
    print("LeafAuthenticityGate initialized successfully.")
    
    # Test on a dummy image
    dummy_img = Image.fromarray(np.uint8(np.zeros((224, 224, 3)) + [30, 150, 30]))
    result = gate.analyze_leaf(dummy_img)
    print("Dummy Test Result:")
    for k, v in result.items():
        if k != "metrics":
            print(f"  {k}: {v}")
