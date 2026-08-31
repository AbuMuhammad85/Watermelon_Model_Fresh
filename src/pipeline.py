import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quality_gate import ImageQualityGate
from src.confidence_gate import ConfidenceGate
from src.knowledge_base import get_disease_info, DISEASE_KNOWLEDGE_BASE
from src.recommendation_engine import RecommendationEngine

class WatermelonDiagnosticPipeline:
    def __init__(self, model_path=None, min_confidence=0.65, min_margin=0.15):
        """
        Initializes the entire watermelon disease diagnostic pipeline.
        
        Args:
            model_path (str, optional): Path to the TFLite model. Defaults to points to model_float16.tflite.
            min_confidence (float): Confidence threshold for the classifier.
            min_margin (float): Decision margin threshold between top 1 and top 2 predictions.
        """
        workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if model_path is None:
            self.model_path = os.path.join(workspace_dir, "checkpoints", "model_float16.tflite")
        else:
            self.model_path = model_path
            
        # Initialize subcomponents
        self.quality_gate = ImageQualityGate()
        self.confidence_gate = ConfidenceGate(min_confidence=min_confidence, min_margin=min_margin)
        self.rec_engine = RecommendationEngine()
        
        # Load TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Classifier class names in alphabetical order (as trained)
        self.classes = ["Anthracnose", "Downy_Mildew", "Healthy", "Mosaic_Virus"]

    def _estimate_severity_proxy(self, pil_image):
        """
        Estimates a heuristic visual severity index (ratio of necrotic/yellow spots to total leaf area)
        using simple color-based segmentation (Excess Green vs Brown/Yellow).
        
        This is clearly flagged as a heuristic proxy rather than a scientific model.
        """
        img_rgb = pil_image.convert("RGB")
        img_np = np.array(img_rgb).astype(np.float32)
        
        r = img_np[:, :, 0]
        g = img_np[:, :, 1]
        b = img_np[:, :, 2]
        
        # Excess Green Index for healthy green tissue
        exg = 2.0 * g - r - b
        green_mask = (exg > 15.0) & (g > 35.0)
        
        # Necrotic/yellow/brown spots heuristic
        # Brown/yellow spots are characterized by R and G being high relative to B, and less excess green
        necrotic_mask = (r > b) & (g > b) & (exg <= 15.0) & (r > 35.0) & (g > 35.0)
        
        green_pixels = np.sum(green_mask)
        necrotic_pixels = np.sum(necrotic_mask)
        total_leaf_pixels = green_pixels + necrotic_pixels
        
        if total_leaf_pixels == 0:
            return 0.0, "Low"
            
        ratio = float(necrotic_pixels / total_leaf_pixels)
        
        # Map ratio to Low/Moderate/High
        if ratio < 0.08:
            label = "Low"
        elif ratio < 0.22:
            label = "Moderate"
        else:
            label = "High"
            
        return ratio, label

    def crop_to_leaf_bbox(self, pil_image, padding=12):
        """
        Crops the image to the bounding box of the green leaf pixels to zoom in on far away leaves
        and reduce surrounding background (soil, weeds) influence.
        """
        img_rgb = pil_image.convert("RGB")
        img_np = np.array(img_rgb).astype(np.float32)
        
        r = img_np[:, :, 0]
        g = img_np[:, :, 1]
        b = img_np[:, :, 2]
        exg = 2.0 * g - r - b
        
        # Use a slightly lower ExG threshold for cropping to make sure we don't miss leaf borders
        green_mask = (exg > 12.0) & (g > 30.0)
        
        # Use binary opening to clean up isolated noise pixels (e.g. soil texture variations)
        from scipy.ndimage import binary_opening
        green_mask_cleaned = binary_opening(green_mask, structure=np.ones((3, 3)))
        
        y_indices, x_indices = np.where(green_mask_cleaned)
        if len(y_indices) == 0 or len(x_indices) == 0:
            # Fallback to uncleaned mask if it was completely emptied
            y_indices, x_indices = np.where(green_mask)
            
        if len(y_indices) == 0 or len(x_indices) == 0:
            return pil_image  # Fallback if no leaf pixels found
            
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        
        h, w, _ = img_np.shape
        ymin = max(0, ymin - padding)
        ymax = min(h, ymax + padding)
        xmin = max(0, xmin - padding)
        xmax = min(w, xmax + padding)
        
        return img_rgb.crop((xmin, ymin, xmax, ymax))

    def run_inference(self, pil_image):
        """
        Performs inference through the TFLite model, after applying leaf crop reduction.
        """
        cropped_img = self.crop_to_leaf_bbox(pil_image)
        
        # Resize to 224x224 and format input data
        img = cropped_img.convert("RGB")
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        input_data = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        
        # Set tensor, invoke, get output
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        return predictions

    def diagnose(self, pil_image, bypass_quality_gate=False, user_severity=None, lang="English"):
        """
        Runs the full diagnostic pipeline on the given image.
        
        Args:
            pil_image (PIL.Image): Watermelon leaf photo.
            bypass_quality_gate (bool): Whether to ignore quality gate warnings.
            user_severity (str, optional): Custom severity ("Low", "Moderate", "High"). If none, uses estimated proxy.
            lang (str): Language for outputs ("English" or "Hausa").
            
        Returns:
            dict: The full diagnostic pipeline results.
        """
        lang = "English" if lang not in ["English", "Hausa"] else lang
        
        # 1. Run Image Quality Gate
        quality_metrics = self.quality_gate.analyze_image(pil_image)
        
        # Gather quality warnings
        quality_warning = ""
        if not quality_metrics["passed"]:
            quality_warning = self.rec_engine.get_quality_warning(quality_metrics["reasons"], lang=lang)
            
        if not quality_metrics["passed"] and not bypass_quality_gate:
            # Short-circuit if quality fails and we don't bypass
            rec_result = self.rec_engine.get_recommendation(
                diagnosis="Unknown",
                confidence_status="LOW_CONFIDENCE",
                lang=lang
            )
            return {
                "quality_passed": False,
                "quality_metrics": quality_metrics,
                "quality_warning": quality_warning,
                "inference_run": False,
                "diagnosis": "Unknown",
                "diagnosis_translated": "Ba a sani ba" if lang == "Hausa" else "Unknown",
                "confidence_status": "LOW_CONFIDENCE",
                "confidence_score": 0.0,
                "predictions_breakdown": {},
                "severity_proxy_ratio": 0.0,
                "severity": "Low",
                "is_severity_scientific": False,
                "recommendation_result": rec_result,
                "disease_info": None,
                "disclaimer": DISEASE_KNOWLEDGE_BASE["Disclaimer"][lang]
            }
            
        # 2. Run TFLite Model Inference
        predictions = self.run_inference(pil_image)
        
        # 3. Evaluate Confidence Gate
        conf_metrics = self.confidence_gate.evaluate_predictions(predictions)
        
        predicted_idx = conf_metrics["top_class_idx"]
        predicted_class_raw = self.classes[predicted_idx]
        confidence_score = conf_metrics["top_confidence"]
        
        # Map confidence status
        confidence_status = "LOW_CONFIDENCE" if conf_metrics["uncertain"] else "HIGH_CONFIDENCE"
        
        # 4. Severity Assessment
        severity_ratio, severity_label = self._estimate_severity_proxy(pil_image)
        
        # If the predicted class is healthy, override severity to Low
        if predicted_class_raw == "Healthy":
            severity_ratio = 0.0
            severity_label = "Low"
            
        # Use user-specified severity if provided
        final_severity = user_severity if user_severity in ["Low", "Moderate", "High"] else severity_label
        
        # 5. Recommendation Engine lookup
        rec_result = self.rec_engine.get_recommendation(
            diagnosis=predicted_class_raw,
            confidence_status=confidence_status,
            severity=final_severity,
            lang=lang
        )
        
        # 6. Disease Knowledge Base lookup
        disease_info = None
        diagnosis_translated = predicted_class_raw
        
        if confidence_status == "HIGH_CONFIDENCE":
            info = get_disease_info(predicted_class_raw)
            if info:
                disease_info = {
                    "name": info["name"][lang],
                    "symptoms": info["symptoms"][lang],
                    "causes": info["causes"][lang],
                    "prevention": info["prevention"][lang],
                    "management": info["management"][lang],
                    "farmer_guidance": info["farmer_guidance"][lang]
                }
                diagnosis_translated = info["name"][lang]
        else:
            diagnosis_translated = "Rashin Tabbaci / Low Confidence" if lang == "Hausa" else "Low Confidence"
            
        # Map breakdown
        breakdown = {self.classes[i]: float(predictions[i]) for i in range(len(self.classes))}
        
        return {
            "quality_passed": quality_metrics["passed"],
            "quality_metrics": quality_metrics,
            "quality_warning": quality_warning,
            "inference_run": True,
            "diagnosis": predicted_class_raw,
            "diagnosis_translated": diagnosis_translated,
            "confidence_status": confidence_status,
            "confidence_score": confidence_score,
            "predictions_breakdown": breakdown,
            "severity_proxy_ratio": severity_ratio,
            "severity": final_severity,
            "is_severity_scientific": False,  # Explicitly stating that this is not scientific
            "recommendation_result": rec_result,
            "disease_info": disease_info,
            "disclaimer": DISEASE_KNOWLEDGE_BASE["Disclaimer"][lang]
        }

if __name__ == "__main__":
    pipeline = WatermelonDiagnosticPipeline()
    print("Pipeline initialized successfully.")
    
    # Test on a blank green image
    dummy_img = Image.fromarray(np.uint8(np.zeros((224, 224, 3)) + [30, 150, 30]))
    res = pipeline.diagnose(dummy_img, bypass_quality_gate=True)
    print("\nDummy Diagnose Result (English):")
    print(f"  Diagnosis: {res['diagnosis']}")
    print(f"  Confidence: {res['confidence_score']:.4f}")
    print(f"  Confidence Status: {res['confidence_status']}")
    print(f"  Severity Label: {res['severity']}")
    print(f"  Recommendation: {res['recommendation_result']['recommendation']}")
