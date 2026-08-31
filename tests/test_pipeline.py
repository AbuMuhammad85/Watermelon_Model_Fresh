import os
import sys
import unittest
import numpy as np
from PIL import Image

# Add root folder to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quality_gate import ImageQualityGate
from src.confidence_gate import ConfidenceGate
from src.knowledge_base import DISEASE_KNOWLEDGE_BASE, get_disease_info
from src.recommendation_engine import RecommendationEngine
from src.pipeline import WatermelonDiagnosticPipeline

class TestWatermelonPipeline(unittest.TestCase):
    
    def setUp(self):
        self.quality_gate = ImageQualityGate()
        self.confidence_gate = ConfidenceGate(min_confidence=0.65, min_margin=0.15)
        self.rec_engine = RecommendationEngine()
        self.pipeline = WatermelonDiagnosticPipeline()

    def test_quality_gate_darkness(self):
        """Test that dark images fail the darkness threshold check."""
        dark_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        metrics = self.quality_gate.analyze_image(dark_img)
        self.assertFalse(metrics["passed"])
        self.assertTrue(metrics["is_dark"])
        self.assertIn("too_dark", metrics["reasons"])

    def test_quality_gate_brightness(self):
        """Test that overexposed images fail the brightness check."""
        bright_img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 250)
        metrics = self.quality_gate.analyze_image(bright_img)
        self.assertFalse(metrics["passed"])
        self.assertTrue(metrics["is_bright"])
        self.assertIn("too_bright", metrics["reasons"])

    def test_quality_gate_blur_flat(self):
        """Test that flat uniform color images (variance = 0) fail the blur check."""
        # Flat grey image
        flat_img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
        metrics = self.quality_gate.analyze_image(flat_img)
        self.assertTrue(metrics["is_blurry"])
        self.assertIn("blurry", metrics["reasons"])

    def test_quality_gate_framing_non_green(self):
        """Test that non-green images fail the framing check."""
        # Pure red image
        red_img = Image.fromarray(np.stack([
            np.ones((224, 224), dtype=np.uint8) * 200,
            np.zeros((224, 224), dtype=np.uint8),
            np.zeros((224, 224), dtype=np.uint8)
        ], axis=-1))
        metrics = self.quality_gate.analyze_image(red_img)
        self.assertTrue(metrics["is_poorly_framed"])
        self.assertIn("leaf_too_small", metrics["reasons"])

    def test_confidence_gate_passing(self):
        """Test that clear high confidence prediction passes the confidence gate."""
        preds = np.array([0.05, 0.85, 0.05, 0.05])
        res = self.confidence_gate.evaluate_predictions(preds)
        self.assertTrue(res["passed"])
        self.assertFalse(res["uncertain"])
        self.assertEqual(res["top_class_idx"], 1)

    def test_confidence_gate_low_confidence(self):
        """Test that low confidence prediction is flagged as uncertain."""
        preds = np.array([0.30, 0.40, 0.20, 0.10])
        res = self.confidence_gate.evaluate_predictions(preds)
        self.assertFalse(res["passed"])
        self.assertTrue(res["uncertain"])
        self.assertEqual(res["reason"], "low_confidence")

    def test_confidence_gate_low_margin(self):
        """Test that two close predictions are flagged as uncertain."""
        preds = np.array([0.05, 0.70, 0.60, 0.05])
        res = self.confidence_gate.evaluate_predictions(preds)
        self.assertFalse(res["passed"])
        self.assertTrue(res["uncertain"])
        self.assertEqual(res["reason"], "low_margin")


    def test_knowledge_base_retrieval(self):
        """Verify the offline knowledge base content and bilingual structures."""
        # Test get_disease_info function
        info = get_disease_info("Anthracnose")
        self.assertIsNotNone(info)
        self.assertIn("English", info["name"])
        self.assertIn("Hausa", info["name"])
        self.assertIn("symptoms", info)
        self.assertIn("causes", info)
        self.assertIn("prevention", info)
        
        # Test Healthy info
        info_healthy = get_disease_info("Healthy")
        self.assertIsNotNone(info_healthy)
        
        # Test disclaimer existence
        self.assertIn("Disclaimer", DISEASE_KNOWLEDGE_BASE)
        self.assertIn("English", DISEASE_KNOWLEDGE_BASE["Disclaimer"])
        self.assertIn("Hausa", DISEASE_KNOWLEDGE_BASE["Disclaimer"])

    def test_recommendation_engine_routing(self):
        """Test that recommendations correctly map by language and severity."""
        # Test low confidence retake recommendation
        rec_low = self.rec_engine.get_recommendation("Anthracnose", "LOW_CONFIDENCE", lang="English")
        self.assertFalse(rec_low["is_actionable"])
        self.assertEqual(rec_low["type"], "retake_request")
        self.assertIn("Low diagnostic confidence", rec_low["recommendation"])

        rec_low_ha = self.rec_engine.get_recommendation("Anthracnose", "LOW_CONFIDENCE", lang="Hausa")
        self.assertIn("Karancin tabbaci", rec_low_ha["recommendation"])

        # Test Healthy recommendation
        rec_healthy = self.rec_engine.get_recommendation("Healthy", "HIGH_CONFIDENCE", severity="Low", lang="English")
        self.assertFalse(rec_healthy["is_actionable"])
        self.assertEqual(rec_healthy["type"], "maintenance")
        self.assertIn("healthy", rec_healthy["recommendation"].lower())

        # Test Anthracnose High severity
        rec_anthracnose_high = self.rec_engine.get_recommendation("Anthracnose", "HIGH_CONFIDENCE", severity="High", lang="English")
        self.assertTrue(rec_anthracnose_high["is_actionable"])
        self.assertEqual(rec_anthracnose_high["type"], "treatment")
        self.assertIn("pull up and destroy", rec_anthracnose_high["recommendation"].lower())

    def test_pipeline_diagnose_structure(self):
        """Verify the full pipeline output dictionary contains all required fields on a leaf."""
        # Load a representative leaf image or synthetic leaf
        leaf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Watermelon", "Healthy", "IMG_2480.jpg")
        if os.path.exists(leaf_path):
            leaf_img = Image.open(leaf_path)
        else:
            # Synthetic leaf with texture
            img_np = np.zeros((224, 224, 3), dtype=np.uint8)
            img_np[:, :, 0] = 40
            img_np[:, :, 1] = 160
            img_np[:, :, 2] = 40
            np.random.seed(42)
            noise = (np.random.rand(224, 224) * 50).astype(np.uint8)
            img_np[:, :, 1] = np.clip(img_np[:, :, 1] + noise, 0, 255)
            leaf_img = Image.fromarray(img_np)
        
        # Run diagnose with quality gate bypassed
        res = self.pipeline.diagnose(leaf_img, bypass_quality_gate=True, lang="English")
        
        # Verify structure
        self.assertIn("quality_passed", res)
        self.assertIn("leaf_gate_passed", res)
        self.assertIn("leaf_gate_result", res)
        self.assertIn("inference_run", res)
        self.assertIn("diagnosis", res)
        self.assertIn("diagnosis_translated", res)
        self.assertIn("confidence_status", res)
        self.assertIn("confidence_score", res)
        self.assertIn("predictions_breakdown", res)
        self.assertIn("severity", res)
        self.assertIn("recommendation_result", res)
        self.assertIn("disclaimer", res)
        
        # Check explicit severity scientific flag is false
        self.assertFalse(res["is_severity_scientific"])

    def test_pipeline_whole_fruit_rejection(self):
        """Verify that a whole watermelon fruit is stopped before inference with clear messaging."""
        from PIL import ImageDraw
        fruit_img = Image.new('RGB', (512, 512), (160, 140, 120))
        draw = ImageDraw.Draw(fruit_img)
        draw.ellipse([56, 56, 456, 456], fill=(60, 130, 50))
        for x in range(80, 440, 35):
            draw.line([(x, 60), (x - 15, 450)], fill=(25, 65, 25), width=16)
            
        res = self.pipeline.diagnose(fruit_img, bypass_quality_gate=True, lang="English")
        
        # Must NOT run model inference
        self.assertFalse(res["inference_run"])
        self.assertFalse(res["leaf_gate_passed"])
        self.assertEqual(res["confidence_status"], "REJECTED")
        self.assertEqual(res["confidence_score"], 0.0)
        self.assertIn("Please capture a clear watermelon leaf, not the watermelon fruit.", res["recommendation_result"]["recommendation"])

    def test_pipeline_whole_fruit_rejection_hausa(self):
        """Verify fruit rejection returns clear Hausa guidance when lang is Hausa."""
        from PIL import ImageDraw
        fruit_img = Image.new('RGB', (512, 512), (160, 140, 120))
        draw = ImageDraw.Draw(fruit_img)
        draw.ellipse([56, 56, 456, 456], fill=(60, 130, 50))
        for x in range(80, 440, 35):
            draw.line([(x, 60), (x - 15, 450)], fill=(25, 65, 25), width=16)
            
        res = self.pipeline.diagnose(fruit_img, bypass_quality_gate=True, lang="Hausa")
        
        self.assertFalse(res["inference_run"])
        self.assertFalse(res["leaf_gate_passed"])
        self.assertEqual(res["confidence_status"], "REJECTED")
        self.assertIn("kankana", res["recommendation_result"]["recommendation"].lower())

    def test_soil_background_processing(self):
        """Test that soil background perturbation runs through pipeline crop step."""
        from src.robustness import perturb_soil_background
        # Create a dummy image with a green square in the center
        img_np = np.zeros((224, 224, 3), dtype=np.uint8)
        img_np[80:144, 80:144, 1] = 180  # green channel
        img_np[80:144, 80:144, 0] = 50   # red channel
        img_np[80:144, 80:144, 2] = 50   # blue channel
        img = Image.fromarray(img_np)
        
        # Apply soil perturbation
        perturbed_img = perturb_soil_background(img)
        
        # Run crop_to_leaf_bbox on it
        cropped = self.pipeline.crop_to_leaf_bbox(perturbed_img)
        self.assertIsInstance(cropped, Image.Image)
        
        # Bounding box should crop out background, so cropped size should be smaller than original
        self.assertLess(cropped.size[0] * cropped.size[1], img.size[0] * img.size[1])

    def test_far_leaf_quality_gate_and_crop(self):
        """Test that a small leaf fails the quality gate and gets zoomed-in by cropping."""
        # Create a far-away leaf image: a tiny green square in a large black background
        img_np = np.zeros((224, 224, 3), dtype=np.uint8)
        img_np[100:115, 100:115, 1] = 180  # very small green region (15x15 = 225 pixels out of 50176 = ~0.4%)
        img_np[100:115, 100:115, 0] = 50
        img_np[100:115, 100:115, 2] = 50
        img = Image.fromarray(img_np)
        
        # Check image quality gate
        metrics = self.quality_gate.analyze_image(img)
        self.assertFalse(metrics["passed"])
        self.assertTrue(metrics["is_poorly_framed"])
        self.assertIn("leaf_too_small", metrics["reasons"])
        
        # Check crop_to_leaf_bbox on this image
        cropped = self.pipeline.crop_to_leaf_bbox(img, padding=5)
        self.assertLessEqual(cropped.size[0], 40)
        self.assertLessEqual(cropped.size[1], 40)

if __name__ == "__main__":
    unittest.main()
