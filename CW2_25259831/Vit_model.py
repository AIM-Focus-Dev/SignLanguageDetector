import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMG_SIZE = 224  
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 8
TRANSFORMER_LAYERS = 8
MLP_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
NUM_CLASSES = 24  # Excluding J and Z
BATCH_SIZE = 32
EPOCHS = 20

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_model():
    # Input layer
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Create patches
    patches = layers.Rescaling(1./255)(inputs)
    patches = layers.Conv2D(
        filters=PROJECTION_DIM,
        kernel_size=PATCH_SIZE,
        strides=PATCH_SIZE,
        padding="valid",
    )(patches)
    patches = layers.Reshape((NUM_PATCHES, PROJECTION_DIM))(patches)
    
    # Encode patches
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    # Create transformer blocks
    for _ in range(TRANSFORMER_LAYERS):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = layers.Dense(MLP_UNITS[0], activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(MLP_UNITS[1], activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # Add MLP
    features = layers.Dense(MLP_UNITS[0], activation=tf.nn.gelu)(representation)
    features = layers.Dropout(0.5)(features)
    
    # Classification head
    logits = layers.Dense(NUM_CLASSES, activation='softmax')(features)

    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def load_data():
    images = []
    labels = []
    
    # Create label mapping for letters A-Y (excluding J and Z)
    letters = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]
    label_mapping = {letter: idx for idx, letter in enumerate(letters)}
    
    print("Label mapping:", label_mapping)  # Debug print
    
    base_path = './data/Dataset/'
    skipped_files = []
    
    for user_folder in os.listdir(base_path):
        user_path = os.path.join(base_path, user_folder)
        if os.path.isdir(user_path):
            for image_file in os.listdir(user_path):
                try:
                    if image_file.endswith('.jpg'):
                        letter = image_file[0].upper()  # First character is the letter, ensure uppercase
                        
                        # Skip J and Z, and any other invalid letters
                        if letter in label_mapping:
                            img_path = os.path.join(user_path, image_file)
                            img = cv2.imread(img_path)
                            if img is not None:  # Check if image was loaded successfully
                                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                                images.append(img)
                                labels.append(label_mapping[letter])
                            else:
                                skipped_files.append(f"Could not read image: {img_path}")
                        else:
                            skipped_files.append(f"Skipped letter {letter} in file: {image_file}")
                except Exception as e:
                    skipped_files.append(f"Error processing {image_file}: {str(e)}")
    
    if skipped_files:
        print("\nSkipped files:")
        for msg in skipped_files[:10]:  # Show first 10 skipped files
            print(msg)
        if len(skipped_files) > 10:
            print(f"... and {len(skipped_files) - 10} more")
    
    if not images:
        raise ValueError("No images were loaded. Please check the dataset directory structure.")
    
    print(f"\nTotal images loaded: {len(images)}")
    print(f"Unique labels found: {sorted(set(labels))}")
    
    X = np.array(images)
    y = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    
    # Print shape information for debugging
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = tf.math.confusion_matrix(
        np.argmax(y_true, axis=1),
        np.argmax(y_pred, axis=1)
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def preprocess_frame(frame):
    """Preprocess camera frame for model prediction"""
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    preprocessed = resized.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    return preprocessed

def predict_letter(prediction):
    """Convert prediction to letter"""
    letters = 'ABCDEFGHIKLMNOPQRSTUVWXY'
    idx = np.argmax(prediction)
    return letters[idx] if 0 <= idx < len(letters) else 'Invalid'

def run_camera_test(model):
    """Run real-time camera test"""
    cap = cv2.VideoCapture(0)
    roi_size = 300
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        height, width = frame.shape[:2]
        x = (width - roi_size) // 2
        y = (height - roi_size) // 2
        
        cv2.rectangle(frame, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
        roi = frame[y:y+roi_size, x:x+roi_size]
        preprocessed = preprocess_frame(roi)
        
        prediction = model.predict(preprocessed, verbose=0)
        letter = predict_letter(prediction)
        confidence = np.max(prediction) * 100
        
        text = f'Prediction: {letter} ({confidence:.1f}%)'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        cv2.imshow('Sign Language Recognition (ViT)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Load and preprocess data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Create and compile model
    print("Creating ViT model...")
    model = create_vit_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test)
    )
    
    # Generate and save training plots
    print("Generating training plots...")
    plot_training_history(history)
    
    # Generate predictions and classification report
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(
        np.argmax(y_test, axis=1),
        np.argmax(y_pred, axis=1),
        target_names=list('ABCDEFGHIKLMNOPQRSTUVWXY')
    ))
    
    # Generate and save confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)
    
    # Save model
    model.save('sign_language_vit_model.h5')
    
    # Run camera test
    print("Starting camera test... Press 'q' to quit")
    run_camera_test(model)

if __name__ == "__main__":
    main()
