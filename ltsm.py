import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Constants
IMG_SIZE = 28
NUM_CLASSES = 26
BATCH_SIZE = 32
EPOCHS = 20
LSTM_UNITS = 256

def create_lstm_model():
    model = models.Sequential([
        # Reshape layer to treat image as sequence of rows
        layers.Reshape((IMG_SIZE, IMG_SIZE), input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        # Bidirectional LSTM layers
        layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True)),
        layers.Dropout(0.3),
        
        layers.Bidirectional(layers.LSTM(LSTM_UNITS)),
        layers.Dropout(0.3),
        
        # Dense layers for classification
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
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

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('LSTM Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('LSTM Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('lstm_training_history.png')
    print("\nTraining history plot saved as 'lstm_training_history.png'")
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('LSTM Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('lstm_confusion_matrix.png')
    print("Confusion matrix plot saved as 'lstm_confusion_matrix.png'")
    plt.close()

def generate_classification_report(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    labels = [chr(i + 65) for i in range(26)]  # A-Z
    
    report = classification_report(y_true, y_pred, target_names=labels, digits=3)
    print("\nClassification Report:")
    print(report)
    
    with open('lstm_classification_report.txt', 'w') as f:
        f.write(report)
    print("Classification report saved as 'lstm_classification_report.txt'")

def preprocess_frame(frame):
    """Preprocess camera frame for model prediction"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padded = cv2.copyMakeBorder(gray, pad_h, pad_h, pad_w, pad_w, 
                               cv2.BORDER_CONSTANT, value=0)
    
    resized = cv2.resize(padded, (IMG_SIZE, IMG_SIZE))
    preprocessed = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    
    return preprocessed

def predict_letter(prediction):
    letters = 'ABCDEFGHIKLMNOPQRSTUVWXY'
    idx = np.argmax(prediction)
    if idx == 9 or idx == 25:  # J and Z require motion
        return 'Invalid (J/Z)'
    return letters[idx] if 0 <= idx < len(letters) else 'Invalid'

def run_camera_test(model):
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
        
        cv2.imshow('Sign Language Recognition (LSTM)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def train_model():
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_data()
    
    # Create model
    model = create_lstm_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test)
    )
    
    # Generate predictions for evaluation
    y_pred = model.predict(X_test)
    
    # Generate visualizations and reports
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    generate_classification_report(y_test, y_pred)
    
    # Save model
    model.save('sign_language_lstm_model.h5')
    
    return model, history

if __name__ == "__main__":
    try:
        print("Loading existing LSTM model...")
        model = tf.keras.models.load_model('sign_language_lstm_model.h5')
        
        # Generate evaluation metrics using test data
        X_train, y_train, X_test, y_test = load_data()
        y_pred = model.predict(X_test)
        
        plot_confusion_matrix(y_test, y_pred)
        generate_classification_report(y_test, y_pred)
        
    except:
        print("Training new LSTM model...")
        model, history = train_model()
    
    print("Starting camera test... Press 'q' to quit")
    run_camera_test(model)
