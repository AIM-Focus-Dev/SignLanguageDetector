import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import cv2
import time

# Hyperparameters
IMG_SIZE = 28
NUM_CLASSES = 26
BATCH_SIZE = 32
EPOCHS = 20

def create_model():
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def load_data():
    # Load data
    train_data = pd.read_csv('./data/sign_mnist_train.csv')
    test_data = pd.read_csv('./data/sign_mnist_test.csv')
    
    # Split features and labels
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values
    
    # Reshape and normalize
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    return X_train, y_train, X_test, y_test

def train_model():
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_data()
    
    # Create model
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test)
    )
    
    # Save model
    model.save('sign_language_model.h5')
    
    return model, history

def preprocess_frame(frame):
    """Preprocess camera frame for model prediction"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Add padding to make it square
    h, w = gray.shape
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padded = cv2.copyMakeBorder(gray, pad_h, pad_h, pad_w, pad_w, 
                               cv2.BORDER_CONSTANT, value=0)
    
    # Resize to model input size
    resized = cv2.resize(padded, (IMG_SIZE, IMG_SIZE))
    
    # Normalize and reshape
    preprocessed = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    
    return preprocessed

def predict_letter(prediction):
    """Convert prediction to letter"""
    letters = 'ABCDEFGHIKLMNOPQRSTUVWXY'
    idx = np.argmax(prediction)
    if idx == 9 or idx == 25:  # J and Z require motion
        return 'Invalid (J/Z)'
    return letters[idx] if 0 <= idx < len(letters) else 'Invalid'

def run_camera_test(model):
    """Run real-time camera test"""
    cap = cv2.VideoCapture(0)
    
    # Region of interest parameters
    roi_size = 300
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Calculate ROI coordinates
        x = (width - roi_size) // 2
        y = (height - roi_size) // 2
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
        
        # Extract and preprocess ROI
        roi = frame[y:y+roi_size, x:x+roi_size]
        preprocessed = preprocess_frame(roi)
        
        # Make prediction
        prediction = model.predict(preprocessed, verbose=0)
        letter = predict_letter(prediction)
        confidence = np.max(prediction) * 100
        
        # Display prediction
        text = f'Prediction: {letter} ({confidence:.1f}%)'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Sign Language Recognition', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if model exists, if not train new one
    try:
        print("Loading existing model...")
        model = tf.keras.models.load_model('sign_language_model.h5')
    except:
        print("Training new model...")
        model, history = train_model()
    
    # Run camera test
    print("Starting camera test... Press 'q' to quit")
    run_camera_test(model)
