import os
import tensorflow as tf

def get_datasets(processed_dir, batch_size=32, img_size=(224, 224)):
    """
    Builds tf.data.Dataset pipelines for train, val, and test.
    Applies training-only data augmentation and optimizes with prefetching.
    """
    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")
    test_dir = os.path.join(processed_dir, "test")
    
    # Determine classes alphabetically (matching Keras convention)
    class_names = sorted(os.listdir(train_dir))
    
    # 1. Load train dataset (shuffle=True)
    # The default color_mode='rgb' forces conversion to RGB and drops alpha channels
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=42
    )
    
    # 2. Load val dataset (shuffle=False)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )
    
    # 3. Load test dataset (shuffle=False)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )
    
    # Moderate augmentation layers
    # Set seed to ensure reproducible augmentations
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
        tf.keras.layers.RandomRotation(0.15, seed=42),
        tf.keras.layers.RandomZoom(0.1, seed=42),
        tf.keras.layers.RandomContrast(0.1, seed=42)
    ])
    
    # Apply augmentations ONLY to the training dataset
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Prefetch datasets for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names
