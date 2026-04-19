import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Basic Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10  # Low number of epochs for simplicity
DATA_DIR = './dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

def build_model(num_classes):
    """
    Builds a simple Convolutional Neural Network (CNN).
    Contains 3 Convolutional layers, ReLU activations, and a final Dense classification layer.
    """
    model = Sequential([
        # Convolutional Layer 1
        tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Convolutional Layer 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Convolutional Layer 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),  # Simple dropout to prevent overfitting
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("====== Plant Disease Classification Training ======")
    print("Normalizing pixel values and initializing data generators...")
    
    # 1. Data Preprocessing (Normalization to 0-1 range)
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Load testing/validation data
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # Do not shuffle to ensure true labels align with predictions
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"Number of target classes found: {num_classes}")
    
    # 2. Build the Model
    model = build_model(num_classes)
    
    print("\nStarting Model Training...")
    # 3. Train the Model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator
    )
    
    # 4. Save the trained model and class indices
    os.makedirs("models", exist_ok=True)
    model.save('models/plant_disease_model.h5')
    print("\nModel saved to models/plant_disease_model.h5")
    
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
        
    print("\nPlotting Accuracy and Loss Graphs...")
    # 5. Output Evaluation: Plot Accuracy & Loss
    plt.figure(figsize=(12, 4))
    
    # Training & Validation Accuracy Graph
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Training & Validation Loss Graph
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_graphs.png')
    print("Training graphs saved as 'training_graphs.png'")
    
    # 6. Evaluation: Print Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_generator.classes, y_pred)
    
    # Visualize confusion matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=train_generator.class_indices.keys(),
                yticklabels=train_generator.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved as 'confusion_matrix.png'")
    print("Training complete!")

if __name__ == '__main__':
    # Fix potential memory leak issue on Apple Silicon or standard Mac OS instances
    import sys
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    main()
