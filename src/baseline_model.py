import tensorflow as tf

def create_model(num_classes=4, input_shape=(224, 224, 3)):
    """
    Creates a MobileNetV3-Small model with a frozen backbone.
    Preprocessing (e.g. scaling) is handled internally by the base model because
    include_preprocessing=True.
    """
    # Instantiate MobileNetV3-Small with frozen pre-trained weights
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        include_preprocessing=True
    )
    
    # Freeze the backbone architecture
    base_model.trainable = False
    
    # Build functional model
    inputs = tf.keras.Input(shape=input_shape)
    
    # Pass inputs through the frozen base model (explicit training=False for batchnorm)
    x = base_model(inputs, training=False)
    
    # Pool features
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Add dropout regularization
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Softmax output layer
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name="Watermelon_MobileNetV3Small_Baseline")
    
    return model

if __name__ == "__main__":
    # Print model summary for structural verification
    model = create_model()
    model.summary()
