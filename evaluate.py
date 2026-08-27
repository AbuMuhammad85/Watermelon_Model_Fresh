import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.dataset import get_datasets

def evaluate_on_dataset(model, dataset, dataset_name, class_names):
    print(f"\nEvaluating on {dataset_name} set...")
    
    y_true = []
    y_pred = []
    
    # Iterate through the dataset and collect predictions
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=-1))
        y_pred.extend(np.argmax(preds, axis=-1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate global metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    
    print(f"\n=== {dataset_name.upper()} RESULTS ===")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1 Score:    {macro_f1:.4f}")
    
    # Classification report (Precision, Recall, F1 per class)
    print("\nPer-Class Metrics:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Confusion matrix
    print("Confusion Matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y_true, y_pred)
    # Print formatted matrix
    col_width = max(len(name) for name in class_names) + 2
    header = " " * col_width + "".join(f"{name:>{col_width}}" for name in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row_str = f"{name:<{col_width}}"
        for val in cm[i]:
            row_str += f"{val:>{col_width}d}"
        print(row_str)
        
    return acc, bal_acc, macro_f1

def main(args):
    checkpoint_path = args.model_path
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at '{checkpoint_path}'. Make sure to train the model first.")
        return
        
    print(f"Loading best saved model from: {checkpoint_path}")
    # Load model (note: weights are compiled)
    model = tf.keras.models.load_model(checkpoint_path)
    
    # Load datasets
    print("Loading datasets...")
    _, val_ds, test_ds, class_names = get_datasets(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        img_size=(224, 224)
    )
    
    # Evaluate Validation set
    evaluate_on_dataset(model, val_ds, "Validation", class_names)
    
    # Evaluate Test set (Locked)
    evaluate_on_dataset(model, test_ds, "Test (Locked)", class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Watermelon Disease Classifier")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.keras", help="Path to best saved model (.keras)")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Path to processed train/val/test data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    
    args = parser.parse_args()
    main(args)
