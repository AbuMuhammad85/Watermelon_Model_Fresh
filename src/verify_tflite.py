import os
import numpy as np
import tensorflow as tf

from src.dataset import get_datasets

def run_tflite_inference(model_path, images):
    """Runs inference on all images using the TFLite Interpreter."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_idx = input_details[0]["index"]
    output_idx = output_details[0]["index"]
    
    preds = []
    for img in images:
        # TFLite expects shape (1, 224, 224, 3) and dtype float32
        input_data = np.expand_dims(img, axis=0).astype(np.float32)
        interpreter.set_tensor(input_idx, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_idx)
        preds.append(output_data[0])
        
    return np.array(preds)

def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keras_path = os.path.join(workspace_dir, "checkpoints", "best_model.keras")
    f32_tflite_path = os.path.join(workspace_dir, "checkpoints", "model_float32.tflite")
    f16_tflite_path = os.path.join(workspace_dir, "checkpoints", "model_float16.tflite")
    processed_dir = os.path.join(workspace_dir, "data", "processed")
    
    print("=== MODEL VERIFICATION & COMPARISON ===")
    
    # 1. Model Sizes
    keras_size = get_file_size_mb(keras_path)
    f32_size = get_file_size_mb(f32_tflite_path)
    f16_size = get_file_size_mb(f16_tflite_path)
    
    print("\n1. Model Sizes (MB):")
    print(f"  - Original Keras model:   {keras_size:.3f} MB")
    print(f"  - TFLite Float32 model:   {f32_size:.3f} MB")
    print(f"  - TFLite Float16 model:   {f16_size:.3f} MB")
    
    # 2. Load representative test images
    print("\nLoading representative images from test split...")
    _, _, test_ds, _ = get_datasets(processed_dir=processed_dir, batch_size=32)
    
    # Extract 20 images
    test_images = []
    for imgs, _ in test_ds.take(1):
        test_images = imgs.numpy()[:20]
        break
        
    print(f"Loaded {len(test_images)} test images for evaluation.")
    
    # 3. Keras predictions
    print("\nRunning inference with original Keras model...")
    keras_model = tf.keras.models.load_model(keras_path)
    keras_preds = keras_model.predict(test_images, verbose=0)
    
    # 4. TFLite predictions
    print("Running inference with TFLite Float32...")
    f32_preds = run_tflite_inference(f32_tflite_path, test_images)
    
    print("Running inference with TFLite Float16...")
    f16_preds = run_tflite_inference(f16_tflite_path, test_images)
    
    # 5. Calculate discrepancies and class identity matching
    keras_classes = np.argmax(keras_preds, axis=-1)
    f32_classes = np.argmax(f32_preds, axis=-1)
    f16_classes = np.argmax(f16_preds, axis=-1)
    
    classes_match_f32 = np.array_equal(keras_classes, f32_classes)
    classes_match_f16 = np.array_equal(keras_classes, f16_classes)
    
    # Float32 differences
    diff_f32 = np.abs(keras_preds - f32_preds)
    mean_diff_f32 = np.mean(diff_f32)
    max_diff_f32 = np.max(diff_f32)
    
    # Float16 differences
    diff_f16 = np.abs(keras_preds - f16_preds)
    mean_diff_f16 = np.mean(diff_f16)
    max_diff_f16 = np.max(diff_f16)
    
    print("\n2. Prediction Discrepancy Statistics (relative to Keras):")
    print("  - TFLite Float32:")
    print(f"    - Mean Absolute Difference: {mean_diff_f32:.2e}")
    print(f"    - Max Absolute Difference:  {max_diff_f32:.2e}")
    print(f"    - Class Predictions Match:  {classes_match_f32} (identical)")
    print("  - TFLite Float16:")
    print(f"    - Mean Absolute Difference: {mean_diff_f16:.2e}")
    print(f"    - Max Absolute Difference:  {max_diff_f16:.2e}")
    print(f"    - Class Predictions Match:  {classes_match_f16} (identical)")
    
    # Verification validation
    print("\n3. Verification Conclusion:")
    if max_diff_f16 < 1e-3 and classes_match_f16:
        print("  - [PASS] Float16 model is extremely close to the Keras baseline and is recommended for mobile deployment.")
    else:
        print("  - [WARNING] Float16 model shows noticeable prediction drift or mismatch. Review conversion options.")

if __name__ == "__main__":
    main()
