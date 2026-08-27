import os
import argparse
import numpy as np
import tensorflow as tf

from src.dataset import get_datasets
from src.baseline_model import create_model

def calculate_class_weights(processed_dir, class_names):
    """
    Automatically calculate balanced class weights from the training split directory.
    Formula: total_samples / (num_classes * class_samples)
    """
    train_dir = os.path.join(processed_dir, "train")
    counts = []
    
    print("\nCalculating class weights from training directory:")
    for cls in class_names:
        cls_path = os.path.join(train_dir, cls)
        count = len(os.listdir(cls_path))
        counts.append(count)
        print(f"  - {cls}: {count} images")
        
    total_samples = sum(counts)
    num_classes = len(class_names)
    
    class_weights = {}
    for i, count in enumerate(counts):
        # Calculate balanced weight
        weight = total_samples / (num_classes * count)
        class_weights[i] = weight
        print(f"  - Weight for class '{class_names[i]}' (index {i}): {weight:.4f}")
        
    return class_weights

def main(args):
    print("Setting up paths and directories...")
    processed_dir = args.processed_dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 1. Load data pipelines
    print("\nLoading datasets...")
    train_ds, val_ds, _, class_names = get_datasets(
        processed_dir=processed_dir,
        batch_size=args.batch_size,
        img_size=(224, 224)
    )
    
    # 2. Calculate class weights
    class_weights = calculate_class_weights(processed_dir, class_names)
    
    # 3. Create model structure (only training classification head)
    print("\nInitializing MobileNetV3-Small baseline model...")
    model = create_model(num_classes=len(class_names))
    
    # Compile model with standard Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # 4. Set up callbacks
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.keras")
    log_path = os.path.join(args.log_dir, "training_log.csv")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=args.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=log_path,
            separator=",",
            append=False
        )
    ]
    
    print("\nCallbacks configured:")
    print(f"  - ModelCheckpoint: saving best model to '{checkpoint_path}'")
    print(f"  - CSVLogger: logging progress to '{log_path}'")
    print(f"  - EarlyStopping: patience={args.early_stopping_patience}")
    print(f"  - ReduceLROnPlateau: patience={args.reduce_lr_patience}")
    
    # 5. Execute training (This is what the user runs manually)
    print("\nStarting manual training process...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small Head on Watermelon Dataset")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Path to processed train/val/test data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save training logs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--reduce_lr_payout_patience", type=int, dest="reduce_lr_patience", default=5, help="Patience for ReduceLROnPlateau")
    
    args = parser.parse_args()
    main(args)
