import os
import numpy as np
import tensorflow as tf

from src.dataset import get_datasets
from src.baseline_model import create_model

def run_tests():
    print("=== STARTING PIPELINE & MODEL VERIFICATION TESTS ===")
    
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(workspace_dir, "data", "processed")
    
    # 1. Test Dataset Loading
    print("\n[TEST 1] Loading datasets from directory...")
    try:
        train_ds, val_ds, test_ds, class_names = get_datasets(
            processed_dir=processed_dir,
            batch_size=32,
            img_size=(224, 224)
        )
        print("  - Successfully loaded datasets.")
        print(f"  - Classes identified: {class_names}")
        assert len(class_names) == 4, f"Expected 4 classes, got {len(class_names)}"
    except Exception as e:
        print(f"  - [FAIL] Error loading datasets: {e}")
        raise e
        
    # 2. Test Batch Shapes & Augmentation
    print("\n[TEST 2] Verifying training batch shapes and augmentations...")
    try:
        # Take one batch from the training dataset
        for images, labels in train_ds.take(1):
            img_shape = images.shape
            lbl_shape = labels.shape
            print(f"  - Train Batch Image shape: {img_shape}")
            print(f"  - Train Batch Label shape: {lbl_shape}")
            
            # Assertions
            assert img_shape == (32, 224, 224, 3), f"Expected (32, 224, 224, 3), got {img_shape}"
            assert lbl_shape == (32, 4), f"Expected (32, 4), got {lbl_shape}"
            
            # Check value range (MobileNetV3 handles preprocessing internally, so raw inputs are [0, 255])
            min_val = np.min(images.numpy())
            max_val = np.max(images.numpy())
            print(f"  - Image value range: [{min_val:.1f}, {max_val:.1f}]")
            assert max_val <= 255.0, f"Expected pixel range within [0, 255], got max {max_val}"
            break
    except Exception as e:
        print(f"  - [FAIL] Error in training batch check: {e}")
        raise e
        
    # 3. Test Model Structure and Backbone Freeze
    print("\n[TEST 3] Initializing and testing MobileNetV3-Small model baseline...")
    try:
        model = create_model(num_classes=4)
        
        # Count parameters
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        
        print(f"  - Model Trainable Parameters: {trainable_count:,}")
        print(f"  - Model Non-Trainable Parameters: {non_trainable_count:,}")
        
        # Verify backbone is frozen (the backbone of MobileNetV3-Small is frozen, only classification head is trainable)
        assert trainable_count > 0, "Classification head should be trainable"
        assert non_trainable_count > 500000, "Backbone weights should be frozen as non-trainable"
        
    except Exception as e:
        print(f"  - [FAIL] Error in model initialization check: {e}")
        raise e
        
    # 4. Test Model Forward Pass (Dry Run)
    print("\n[TEST 4] Running a dry-run inference on a batch of images...")
    try:
        # Pass the batch of images through the model
        preds = model(images, training=False)
        print(f"  - Output predictions batch shape: {preds.shape}")
        assert preds.shape == (32, 4), f"Expected output predictions shape (32, 4), got {preds.shape}"
        
        # Check softmax properties (sums to 1)
        sums = np.sum(preds.numpy(), axis=-1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)
        print("  - Prediction softmax sums verified to be 1.0.")
        
    except Exception as e:
        print(f"  - [FAIL] Error in forward pass inference check: {e}")
        raise e
        
    print("\n=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ===")
    return True

if __name__ == "__main__":
    run_tests()
