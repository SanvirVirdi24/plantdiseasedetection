import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os
import json
import random
import logging

# Suppress TF warnings for a cleaner output
tf.get_logger().setLevel(logging.ERROR)

IMG_HEIGHT = 128
IMG_WIDTH = 128

def load_labels_and_model():
    """Loads the pre-trained plant disease model and class indices."""
    if not os.path.exists('models/plant_disease_model.h5'):
        print("Error: Model not found. Please run train.py first to generate the model.")
        sys.exit(1)
        
    model = tf.keras.models.load_model('models/plant_disease_model.h5')
    
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
        
    # Create an inverse mapping from index to class name
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

def segment_leaf(img_rgb):
    """
    Performs simple leaf segmentation using OpenCV color thresholding.
    Extracts the green parts of the leaf for simple background removal visualization.
    """
    # Convert image from RGB space to HSV color space
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Define range of green color in HSV space
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    
    # Threshold the HSV image to get only green colors (mask)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Bitwise-AND the mask with the original image to get the segmented leaf
    segmented_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    return segmented_img, mask

def predict_and_display(image_path):
    """
    Loads an image, segments it, predicts the disease using the CNN model, 
    and displays the final results side-by-side.
    """
    model, labels = load_labels_and_model()

    print(f"\nProcessing Image: {image_path}")
    
    # 1. Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}. Ensure the file exists and is valid.")
        return
        
    # OpenCV loads images in BGR format, convert to RGB for standard display/processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Perform leaf segmentation
    segmented_img, mask = segment_leaf(img_rgb)
    
    # 3. Preprocess for Model Prediction (resize & normalize)
    # The CNN model expects 128x128 images (just like during training)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized / 255.0  # Normalize pixels from 0-255 to [0, 1]
    
    # Expand dims to represent a single batch (Model expects shapes like (1, 128, 128, 3))
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Run inference
    predictions = model.predict(img_batch)
    
    # Get top prediction index and mapping back to string class label
    top_pred_index = np.argmax(predictions[0])
    confidence = predictions[0][top_pred_index] * 100
    predicted_label = labels[top_pred_index]
    
    print(f"==> Predicted Output: {predicted_label}")
    print(f"==> Confidence Level: {confidence:.2f}%")
    
    # 4. Display the Results
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Leaf Image')
    plt.axis('off')
    
    # Subplot 2: Segmented Mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('HSV Color Segmentation Mask')
    plt.axis('off')
    
    # Subplot 3: Segmented Output + Prediction Text
    plt.subplot(1, 3, 3)
    plt.imshow(segmented_img)
    plt.title(f'Segmented Output\nPred: {predicted_label} ({confidence:.1f}%)')
    plt.axis('off')
    
    plt.tight_layout()
    output_display = 'prediction_output.png'
    plt.savefig(output_display)
    print(f"\nVisualization saved successfully as '{output_display}'!")
    # Optionally open window if running interactively
    # plt.show() 

if __name__ == '__main__':
    # Handle command-line arguments to allow passing an image directly
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        # If no image path provided, try to pick a random test image from our dataset
        test_dir = './dataset/test'
        if os.path.exists(test_dir):
            classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            if not classes:
                print("No test categories found in the dataset directory.")
                sys.exit(1)
            
            random_class = random.choice(classes)
            class_dir = os.path.join(test_dir, random_class)
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if images:
                random_image = random.choice(images)
                test_img_path = os.path.join(class_dir, random_image)
                print(f"No custom image argument provided. Picking a random test image...")
                predict_and_display(test_img_path)
            else:
                print("No test images found to run predictions on.")
        else:
            print("Test directory does not exist. Cannot default to a random image.")
    else:
        # Using specified image path
        image_path = sys.argv[1]
        predict_and_display(image_path)
