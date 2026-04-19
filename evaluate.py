import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score
import logging

# Suppress TF logs
tf.get_logger().setLevel(logging.ERROR)

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
TEST_DIR = './dataset/test'

def evaluate_model():
    print("Loading model for evaluation...")
    if not os.path.exists('models/plant_disease_model.h5'):
        print("Error: Model not found. Train the model first.")
        return
        
    model = tf.keras.models.load_model('models/plant_disease_model.h5')
    
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
        
    # Reverse dict to get class names
    target_names = {v: k for k, v in class_indices.items()}
    # List in sorted order
    class_names = [target_names[i] for i in range(len(target_names))]
    
    print("Loading test dataset...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print("\nRunning predictions on test set... This might take a minute.")
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION STATISTICS")
    print("="*60)
    
    acc = accuracy_score(test_generator.classes, y_pred)
    print(f"\n✅ OVERALL ACCURACY: {acc*100:.2f}%")
    
    print("\n--- Detailed Classification Report ---")
    report = classification_report(test_generator.classes, y_pred, target_names=class_names)
    print(report)
    print("="*60)
    
    # Save the report to a text file for records
    with open("model_stats.txt", "w") as f:
        f.write("MODEL EVALUATION STATISTICS\n")
        f.write("="*60 + "\n")
        f.write(f"OVERALL ACCURACY: {acc*100:.2f}%\n\n")
        f.write("--- Detailed Classification Report ---\n")
        f.write(report)
        
    print("These statistics have been saved to 'model_stats.txt' in your folder.")

if __name__ == '__main__':
    evaluate_model()
