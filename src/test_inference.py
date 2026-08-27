import os
import numpy as np
import tensorflow as tf
from PIL import Image

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(workspace_dir, "checkpoints", "model_float16.tflite")
    test_dir = os.path.join(workspace_dir, "data", "processed", "test")
    
    classes = ["Anthracnose", "Downy_Mildew", "Healthy", "Mosaic_Virus"]
    
    print("=== AUTOMATED TFLITE FLOAT16 INFERENCE TEST ===")
    print(f"Loading TFLite Float16 model from: {model_path}")
    
    # 1. Initialize Interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"  - [FAIL] Failed to load/initialize Interpreter: {e}")
        return False
        
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_idx = input_details[0]["index"]
    output_idx = output_details[0]["index"]
    
    # 2. Iterate and test one image from each class
    all_passed = True
    for cls in classes:
        print(f"\nTesting class: {cls}")
        class_path = os.path.join(test_dir, cls)
        
        if not os.path.isdir(class_path):
            print(f"  - [FAIL] Directory not found for class: {class_path}")
            all_passed = False
            continue
            
        files = os.listdir(class_path)
        if not files:
            print(f"  - [FAIL] No test images found in directory: {class_path}")
            all_passed = False
            continue
            
        # Select the first image
        img_name = files[0]
        img_path = os.path.join(class_path, img_name)
        print(f"  - Loaded image: {img_name}")
        
        try:
            # Preprocess image to match training pipeline (RGB, 224x224, float32, range [0.0, 255.0])
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224), Image.Resampling.BILINEAR)
            input_data = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
            
            # Run TFLite inference
            interpreter.set_tensor(input_idx, input_data)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_idx)
            
            # Assertions
            print(f"  - Output shape: {preds.shape}")
            assert preds.shape == (1, 4), f"Expected shape (1, 4), got {preds.shape}"
            
            prob_sum = np.sum(preds[0])
            print(f"  - Sum of probabilities: {prob_sum:.6f}")
            np.testing.assert_allclose(prob_sum, 1.0, rtol=1e-5)
            
            pred_class_idx = np.argmax(preds[0])
            print(f"  - Predictions: " + ", ".join(f"{c}: {p*100:.2f}%" for c, p in zip(classes, preds[0])))
            print(f"  - Predicted class index: {pred_class_idx} ({classes[pred_class_idx]})")
            print(f"  - [PASS] Inference successful for {cls}")
            
        except Exception as e:
            print(f"  - [FAIL] Error running inference on {img_name}: {e}")
            all_passed = False
            
    if all_passed:
        print("\n=== ALL INFERENCE TESTS PASSED SUCCESSFULLY! ===")
        return True
    else:
        print("\n=== SOME INFERENCE TESTS FAILED! ===")
        return False

if __name__ == "__main__":
    main()
