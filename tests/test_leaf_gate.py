import os
import glob
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw

# Add root folder to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.leaf_gate import LeafAuthenticityGate

class TestLeafAuthenticityGate(unittest.TestCase):
    
    def setUp(self):
        self.leaf_gate = LeafAuthenticityGate()
        self.workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _create_synthetic_melon(self, rx=200, ry=200, melon_color=(60, 130, 50), stripe_color=(25, 65, 25)):
        """Helper to create a synthetic whole watermelon fruit image."""
        img = Image.new('RGB', (512, 512), (160, 140, 120))  # soil-like background
        draw = ImageDraw.Draw(img)
        draw.ellipse([256 - rx, 256 - ry, 256 + rx, 256 + ry], fill=melon_color)
        for x in range(256 - rx + 20, 256 + rx, 35):
            draw.line([(x, 256 - ry + 10), (x - 15, 256 + ry - 10)], fill=stripe_color, width=14)
        return img

    def test_whole_fruit_rejection_striped_round(self):
        """Test that a round striped whole watermelon fruit is rejected."""
        melon = self._create_synthetic_melon(rx=200, ry=200)
        res = self.leaf_gate.analyze_leaf(melon)
        self.assertFalse(res["is_leaf"])
        self.assertEqual(res["reason"], "fruit_detected")
        self.assertEqual(res["message"], "Please capture a clear watermelon leaf, not the watermelon fruit.")
        self.assertIn("kankana", res["message_hausa"])

    def test_whole_fruit_rejection_oblong(self):
        """Test that an oblong / oval watermelon fruit is rejected."""
        melon = self._create_synthetic_melon(rx=240, ry=160)
        res = self.leaf_gate.analyze_leaf(melon)
        self.assertFalse(res["is_leaf"])
        self.assertEqual(res["reason"], "fruit_detected")
        self.assertEqual(res["message"], "Please capture a clear watermelon leaf, not the watermelon fruit.")

    def test_whole_fruit_rejection_dark_solid(self):
        """Test that a dark solid whole watermelon fruit is rejected."""
        melon = self._create_synthetic_melon(rx=210, ry=190, melon_color=(35, 80, 30), stripe_color=(20, 50, 20))
        res = self.leaf_gate.analyze_leaf(melon)
        self.assertFalse(res["is_leaf"])
        self.assertEqual(res["reason"], "fruit_detected")

    def test_whole_fruit_rejection_light_green(self):
        """Test that a light-green / Charleston-gray style watermelon fruit is rejected."""
        melon = self._create_synthetic_melon(rx=220, ry=180, melon_color=(100, 160, 80), stripe_color=(45, 95, 40))
        res = self.leaf_gate.analyze_leaf(melon)
        self.assertFalse(res["is_leaf"])
        self.assertEqual(res["reason"], "fruit_detected")

    def test_non_leaf_rejection_blank_white(self):
        """Test that a plain white image is rejected."""
        white_img = Image.new('RGB', (512, 512), (255, 255, 255))
        res = self.leaf_gate.analyze_leaf(white_img)
        self.assertFalse(res["is_leaf"])
        self.assertEqual(res["reason"], "insufficient_vegetation")

    def test_non_leaf_rejection_desk(self):
        """Test that a wooden desk surface is rejected."""
        desk_img = Image.new('RGB', (512, 512), (210, 180, 140))
        res = self.leaf_gate.analyze_leaf(desk_img)
        self.assertFalse(res["is_leaf"])

    def test_non_leaf_rejection_blue_sky(self):
        """Test that sky/blue background is rejected."""
        sky_img = Image.new('RGB', (512, 512), (100, 160, 240))
        res = self.leaf_gate.analyze_leaf(sky_img)
        self.assertFalse(res["is_leaf"])
        self.assertEqual(res["reason"], "insufficient_vegetation")

    def test_structured_result_format(self):
        """Test that the result dictionary contains all specified fields."""
        dummy_img = Image.new('RGB', (512, 512), (200, 200, 200))
        res = self.leaf_gate.analyze_leaf(dummy_img)
        
        self.assertIn("is_leaf", res)
        self.assertIn("score", res)
        self.assertIn("reason", res)
        self.assertIn("message", res)
        self.assertIn("message_hausa", res)
        self.assertIn("metrics", res)
        
        metrics = res["metrics"]
        self.assertIn("plant_ratio", metrics)
        self.assertIn("solidity", metrics)
        self.assertIn("extent", metrics)
        self.assertIn("boundary_complexity", metrics)
        self.assertIn("internal_edge_density", metrics)
        self.assertIn("fruit_resemblance", metrics)
        self.assertIn("leaf_authenticity_score", metrics)

    def test_dataset_leaves_anthracnose(self):
        """Test representative Anthracnose leaf samples pass the authenticity gate."""
        imgs = glob.glob(os.path.join(self.workspace_dir, "Watermelon", "Anthracnose", "*.jpg"))[:5]
        if imgs:
            for p in imgs:
                im = Image.open(p)
                res = self.leaf_gate.analyze_leaf(im)
                self.assertTrue(res["is_leaf"], f"Authentic Anthracnose leaf {os.path.basename(p)} was rejected")
                self.assertGreaterEqual(res["score"], self.leaf_gate.min_leaf_score)

    def test_dataset_leaves_downy_mildew(self):
        """Test representative Downy Mildew leaf samples pass the authenticity gate."""
        imgs = glob.glob(os.path.join(self.workspace_dir, "Watermelon", "Downy_Mildew", "*.jpg"))[:5]
        if imgs:
            for p in imgs:
                im = Image.open(p)
                res = self.leaf_gate.analyze_leaf(im)
                self.assertTrue(res["is_leaf"], f"Authentic Downy Mildew leaf {os.path.basename(p)} was rejected")
                self.assertGreaterEqual(res["score"], self.leaf_gate.min_leaf_score)

    def test_dataset_leaves_healthy(self):
        """Test representative Healthy leaf samples pass the authenticity gate."""
        imgs = glob.glob(os.path.join(self.workspace_dir, "Watermelon", "Healthy", "*.jpg"))[:5]
        if imgs:
            for p in imgs:
                im = Image.open(p)
                res = self.leaf_gate.analyze_leaf(im)
                self.assertTrue(res["is_leaf"], f"Authentic Healthy leaf {os.path.basename(p)} was rejected")
                self.assertGreaterEqual(res["score"], self.leaf_gate.min_leaf_score)

    def test_dataset_leaves_mosaic_virus(self):
        """Test representative Mosaic Virus leaf samples pass the authenticity gate."""
        imgs = glob.glob(os.path.join(self.workspace_dir, "Watermelon", "Mosaic_Virus", "*.jpg"))[:5]
        if imgs:
            for p in imgs:
                im = Image.open(p)
                res = self.leaf_gate.analyze_leaf(im)
                self.assertTrue(res["is_leaf"], f"Authentic Mosaic Virus leaf {os.path.basename(p)} was rejected")
                self.assertGreaterEqual(res["score"], self.leaf_gate.min_leaf_score)

if __name__ == "__main__":
    unittest.main()
