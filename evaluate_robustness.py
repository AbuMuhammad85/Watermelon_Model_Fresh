import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import WatermelonDiagnosticPipeline
from src.robustness import (
    perturb_brightness,
    perturb_blur,
    perturb_distance,
    perturb_soil_background
)

def load_test_images(test_dir):
    """
    Loads all images and their labels from the test directory.
    """
    classes = ["Anthracnose", "Downy_Mildew", "Healthy", "Mosaic_Virus"]
    dataset = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"Warning: Class directory {class_path} not found.")
            continue
            
        for file_name in os.listdir(class_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(class_path, file_name)
                dataset.append({
                    "path": img_path,
                    "label_idx": class_idx,
                    "label_name": class_name,
                    "file_name": file_name
                })
                
    return dataset

def main():
    print("====================================================")
    print("      WATERMELON PIPELINE ROBUSTNESS EVALUATION      ")
    print("====================================================\n")
    
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(workspace_dir, "data", "processed", "test")
    log_dir = os.path.join(workspace_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at: {test_dir}")
        print("Please check your data/processed/test structure.")
        return
        
    # Load pipeline
    print("Initializing offline diagnostic pipeline...")
    pipeline = WatermelonDiagnosticPipeline()
    classes = pipeline.classes
    
    # Load test images
    test_dataset = load_test_images(test_dir)
    total_images = len(test_dataset)
    print(f"Loaded {total_images} test images from split.\n")
    
    if total_images == 0:
        print("No test images found. Exiting.")
        return
        
    # Perturbations to evaluate
    perturbations = {
        "Baseline (Clean)": lambda img: img,
        "Low Light (0.4x Brightness)": lambda img: perturb_brightness(img, 0.4),
        "Bright Glare (1.6x Brightness)": lambda img: perturb_brightness(img, 1.6),
        "Out of Focus (Gaussian Blur)": lambda img: perturb_blur(img, 2),
        "Far Away Leaf (0.5x Scale)": lambda img: perturb_distance(img, 0.5),
        "Soil Background Clutter": lambda img: perturb_soil_background(img)
    }
    
    results = {}
    for p_name in perturbations.keys():
        results[p_name] = {
            "correct": 0,
            "confidences": [],
            "class_correct": {c: 0 for c in classes},
            "class_total": {c: 0 for c in classes},
            "mismatches": []  # Track interesting failures
        }
        
    print("Evaluating robustness. Running TFLite inference...")
    
    for idx, item in enumerate(test_dataset):
        if (idx + 1) % 20 == 0 or idx == 0 or idx + 1 == total_images:
            print(f"  Processing image {idx + 1}/{total_images}...")
            
        true_idx = item["label_idx"]
        true_name = item["label_name"]
        
        try:
            raw_img = Image.open(item["path"]).convert("RGB")
        except Exception as e:
            print(f"    Failed to load {item['file_name']}: {e}")
            continue
            
        for p_name, p_func in perturbations.items():
            try:
                # Apply perturbation
                p_img = p_func(raw_img)
                
                # Run prediction directly via TFLite (bypassing quality gate to test raw classifier robustness)
                preds = pipeline.run_inference(p_img)
                pred_idx = np.argmax(preds)
                pred_conf = preds[pred_idx]
                pred_name = classes[pred_idx]
                
                # Track statistics
                is_correct = (pred_idx == true_idx)
                
                results[p_name]["class_total"][true_name] += 1
                if is_correct:
                    results[p_name]["correct"] += 1
                    results[p_name]["class_correct"][true_name] += 1
                else:
                    # Save a few examples of soil background failures
                    if p_name == "Soil Background Clutter" and len(results[p_name]["mismatches"]) < 5:
                        results[p_name]["mismatches"].append({
                            "file": item["file_name"],
                            "true": true_name,
                            "pred": pred_name,
                            "conf": pred_conf
                        })
                        
                results[p_name]["confidences"].append(pred_conf)
                
            except Exception as e:
                print(f"Error evaluating {p_name} on {item['file_name']}: {e}")
                
    # Generate report
    report_lines = []
    report_lines.append("====================================================")
    report_lines.append("             ROBUSTNESS AUDIT REPORT                ")
    report_lines.append("====================================================\n")
    report_lines.append(f"Evaluated {total_images} images across 4 classes.\n")
    
    # Summary Table
    report_lines.append(f"{'Perturbation Scenario':<35} | {'Accuracy':<10} | {'Avg Confidence':<15}")
    report_lines.append("-" * 66)
    
    for p_name in perturbations.keys():
        total_p = sum(results[p_name]["class_total"].values())
        if total_p == 0:
            continue
        acc = results[p_name]["correct"] / total_p
        avg_conf = np.mean(results[p_name]["confidences"]) if results[p_name]["confidences"] else 0.0
        report_lines.append(f"{p_name:<35} | {acc:.2%}    | {avg_conf:.2%}")
        
    report_lines.append("\n====================================================")
    report_lines.append("        BACKGROUND/SOIL CLUTTER INVESTIGATION       ")
    report_lines.append("====================================================\n")
    
    # Compare baseline vs soil
    total_b = sum(results["Baseline (Clean)"]["class_total"].values())
    acc_clean = results["Baseline (Clean)"]["correct"] / total_b if total_b > 0 else 0
    total_s = sum(results["Soil Background Clutter"]["class_total"].values())
    acc_soil = results["Soil Background Clutter"]["correct"] / total_s if total_s > 0 else 0
    
    diff = acc_clean - acc_soil
    
    report_lines.append(f"Baseline Accuracy (Clean):     {acc_clean:.2%}")
    report_lines.append(f"Soil Background Accuracy:      {acc_soil:.2%}")
    report_lines.append(f"Accuracy Degradation Delta:    {diff:+.2%}")
    
    if diff > 0.05:
        report_lines.append("\n[!] WARNING: Soil background causes noticeable degradation in classification accuracy.")
        report_lines.append("    The model is sensitive to background textures. Integrating the framing/leaf area quality gate")
        report_lines.append("    and warning the farmer when soil is dominant is strongly justified.")
    else:
        report_lines.append("\n[PASS] The model exhibits strong robustness against soil and non-leaf background textures.")
        report_lines.append("    The classification accuracy was not significantly degraded by background substitution.")
        
    # Detail soil mismatches
    if results["Soil Background Clutter"]["mismatches"]:
        report_lines.append("\nDetailed Soil Clutter Mismatch Examples:")
        for idx, m in enumerate(results["Soil Background Clutter"]["mismatches"]):
            report_lines.append(f"  {idx+1}. File: {m['file']} | True: {m['true']} | Predicted: {m['pred']} (Conf: {m['conf']:.2%})")
            
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save report
    report_path = os.path.join(log_dir, "robustness_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nRobustness report saved to: {report_path}")

if __name__ == "__main__":
    main()
