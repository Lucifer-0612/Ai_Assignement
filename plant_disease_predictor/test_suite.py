"""
Comprehensive Verification and Test Suite for PhytoScan AI
===========================================================
Executes end-to-end unit, functional, and headless integration tests on:
1. Requirements & Dependencies
2. Core CV & ML Engine (engine.py)
3. Procedural Sample Leaf Generator
4. Diagnostic Classification & Lesion Heatmap Generation
5. Tkinter Desktop GUI (app.py)
6. Documentation & Viva Guide (README.md)
"""

from __future__ import annotations

import os
import sys
import unittest
import tkinter as tk
from pathlib import Path
from PIL import Image

# Ensure plant_disease_predictor is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from engine import (
    PlantDiseaseEngine,
    PredictionResult,
    DiseaseProfile,
    generate_sample_leaf_images,
    DISEASE_DATABASE,
    get_engine,
)
from app import PlantDiseaseApp


class TestPhytoScanAI(unittest.TestCase):
    """End-to-End Test Suite for PhytoScan AI."""

    @classmethod
    def setUpClass(cls):
        cls.engine = get_engine()
        cls.samples_dir = PROJECT_DIR / "sample_leaves"
        cls.sample_paths = generate_sample_leaf_images(str(cls.samples_dir))

    def test_01_requirements_file(self):
        """Verify requirements.txt exists and contains required dependencies."""
        req_file = PROJECT_DIR / "requirements.txt"
        self.assertTrue(req_file.exists(), "requirements.txt missing!")
        content = req_file.read_text(encoding="utf-8")
        self.assertIn("numpy", content)
        self.assertIn("Pillow", content)
        self.assertIn("scikit-learn", content)

    def test_02_sample_leaf_generation(self):
        """Verify all 7 synthetic leaf images are generated as 400x400 PNGs."""
        expected_keys = [
            "healthy",
            "cercospora_beet",
            "early_blight",
            "powdery_mildew",
            "yellow_leaf_curl",
            "bacterial_spot",
            "late_blight",
        ]
        for key in expected_keys:
            self.assertIn(key, self.sample_paths)
            img_path = Path(self.sample_paths[key])
            self.assertTrue(img_path.exists(), f"Sample image {key} does not exist at {img_path}")
            with Image.open(img_path) as img:
                self.assertEqual(img.size, (400, 400), f"Image {key} is not 400x400!")
                self.assertEqual(img.mode, "RGB", f"Image {key} is not RGB!")

    def test_03_engine_diagnostic_accuracy(self):
        """Verify engine classifies each of the 7 sample leaves correctly."""
        for key, path in self.sample_paths.items():
            result = self.engine.analyze_image(path)
            self.assertIsInstance(result, PredictionResult)
            self.assertEqual(
                result.predicted_disease.disease_id,
                key,
                f"Expected class '{key}' but got '{result.predicted_disease.disease_id}' for {path}",
            )
            self.assertGreaterEqual(result.confidence_pct, 60.0, f"Confidence too low for {key}")
            self.assertLess(result.processing_time_ms, 200.0, f"Processing too slow for {key}")
            self.assertIsInstance(result.segmented_overlay_image, Image.Image)
            self.assertEqual(result.segmented_overlay_image.size, (400, 400))

    def test_04_knowledge_base_integrity(self):
        """Verify disease database contains all profiles with treatment plans."""
        self.assertEqual(len(DISEASE_DATABASE), 7)
        for d_id, profile in DISEASE_DATABASE.items():
            self.assertIsInstance(profile, DiseaseProfile)
            self.assertTrue(len(profile.name) > 5)
            self.assertTrue(len(profile.symptoms) > 20)
            self.assertGreaterEqual(len(profile.organic_treatment), 2)
            self.assertGreaterEqual(len(profile.chemical_treatment), 2)
            self.assertGreaterEqual(len(profile.prevention_tips), 2)

    def test_05_gui_headless_execution(self):
        """Verify Tkinter desktop application initializes, updates, and renders cleanly."""
        root = tk.Tk()
        root.withdraw()
        try:
            app = PlantDiseaseApp(root)
            self.assertIsNotNone(app.engine)
            self.assertIsNotNone(app.canvas_orig)
            self.assertIsNotNone(app.canvas_heat)

            # Trigger sample load (index 1: Cercospora)
            app.combo_sample.current(1)
            app._on_sample_selected(None)

            # Run diagnosis worker
            app._diagnostic_worker()
            root.update_idletasks()
            root.update()

            # Verify UI state
            self.assertIn("Cercospora", app.lbl_diag_disease.cget("text"))
            self.assertEqual(len(app.tree_top3.get_children()), 3)
            self.assertEqual(len(app.tree_metrics.get_children()), 5)

            # Verify report generation
            report = app._generate_report_text(app._last_result)
            self.assertIn("PHYTOSCAN AI", report)
            self.assertIn("Cercospora", report)
            self.assertGreater(len(report), 1000)

            # Test reset
            app._on_reset()
            self.assertEqual(len(app.tree_top3.get_children()), 0)
        finally:
            root.destroy()

    def test_06_readme_completeness(self):
        """Verify README.md contains viva questions, architecture, and installation guides."""
        readme = PROJECT_DIR / "README.md"
        self.assertTrue(readme.exists(), "README.md missing!")
        text = readme.read_text(encoding="utf-8")
        self.assertIn("PhytoScan AI", text)
        self.assertIn("Cercospora", text)
        self.assertIn("flowchart TD", text)
        for q_num in range(1, 11):
            self.assertIn(f"Q{q_num}:", text, f"Viva Question Q{q_num} missing from README.md!")

    def test_07_user_specimen_diagnosis(self):
        """Verify the user specimen gettyimages-1901703916-612x612.jpg diagnoses as Cercospora Leaf Spot."""
        specimen_path = Path(r"C:\Users\itsme\Downloads\gettyimages-1901703916-612x612.jpg")
        if specimen_path.exists():
            result = self.engine.analyze_image(str(specimen_path))
            self.assertEqual(
                result.predicted_disease.disease_id,
                "cercospora_beet",
                f"Specimen expected cercospora_beet but got {result.predicted_disease.disease_id}",
            )
            self.assertGreaterEqual(result.confidence_pct, 90.0)
            self.assertIn("Cercospora", result.predicted_disease.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
