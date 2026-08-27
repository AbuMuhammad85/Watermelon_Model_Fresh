import os
import tensorflow as tf

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keras_model_path = os.path.join(workspace_dir, "checkpoints", "best_model.keras")
    
    if not os.path.exists(keras_model_path):
        print(f"Error: Keras model not found at '{keras_model_path}'. Please ensure it has been trained.")
        return
        
    print(f"Loading Keras model from: {keras_model_path}")
    model = tf.keras.models.load_model(keras_model_path)
    
    # 1. Convert to Float32 TFLite
    print("\nConverting model to TFLite Float32...")
    converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_f32_data = converter_f32.convert()
    
    f32_output_path = os.path.join(workspace_dir, "checkpoints", "model_float32.tflite")
    with open(f32_output_path, "wb") as f:
        f.write(tflite_f32_data)
    print(f"Float32 TFLite model saved to: {f32_output_path}")
    
    # 2. Convert to Float16 TFLite
    print("\nConverting model to TFLite Float16...")
    converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_f16.target_spec.supported_types = [tf.float16]
    tflite_f16_data = converter_f16.convert()
    
    f16_output_path = os.path.join(workspace_dir, "checkpoints", "model_float16.tflite")
    with open(f16_output_path, "wb") as f:
        f.write(tflite_f16_data)
    print(f"Float16 TFLite model saved to: {f16_output_path}")
    
    print("\nConversion successfully completed!")

if __name__ == "__main__":
    main()
