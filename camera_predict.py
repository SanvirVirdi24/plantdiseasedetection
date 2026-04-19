import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import json
import logging

# Suppress warnings
tf.get_logger().setLevel(logging.ERROR)

DISEASE_DETAILS = {
    'Pepper__bell___Bacterial_spot': "A bacterial disease that causes water-soaked spots on leaves and fruit. Treatment: Use copper-based bactericides.",
    'Pepper__bell___healthy': "The plant is healthy! Keep up the good work.",
    'Potato___Early_blight': "Fungal disease causing dark lesions with concentric rings. Treatment: Needs proper fungicide.",
    'Potato___Late_blight': "Highly destructive fungal disease. Treatment: Destroy infected plants, apply specific fungicide.",
    'Potato___healthy': "The plant is healthy! Keep up the good work.",
    'Tomato_Bacterial_spot': "Small water-soaked spots. Treatment: Practice crop rotation and use copper sprays.",
    'Tomato_Early_blight': "Causes 'bullseye' spots on lower leaves. Treatment: Treat with protective fungicides.",
    'Tomato_Late_blight': "Causes grey/brown lesions. Extremely contagious. Treatment: Destroy infected plants.",
    'Tomato_Leaf_Mold': "Causes pale green spots and velvety mold. Treatment: Improve air circulation.",
    'Tomato_Septoria_leaf_spot': "Small circular spots with grey centers. Treatment: Remove infected leaves, use fungicide.",
    'Tomato_Spider_mites_Two_spotted_spider_mite': "Tiny pests causing yellow stippling. Treatment: Use miticides or neem oil.",
    'Tomato__Target_Spot': "Fungal pathogen causing brown spots. Treatment: Ensure proper spacing, apply fungicide.",
    'Tomato__Tomato_YellowLeaf__Curl_Virus': "Viral disease from whiteflies. Treatment: Control the whitefly population.",
    'Tomato__Tomato_mosaic_virus': "Viral disease causing mosaic patterns. Treatment: Destroy infected plants immediately.",
    'Tomato_healthy': "The plant is healthy! Keep up the good work."
}

def load_labels_and_model():
    if not os.path.exists('models/plant_disease_model.h5'):
        print("Error: Model not found. Please run train.py first.")
        sys.exit(1)
    
    model = tf.keras.models.load_model('models/plant_disease_model.h5')
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

def display_webcam():
    print("Opening Webcam Window...")
    print(" > Press 'SPACEBAR' to capture photo!")
    print(" > Press 'ESC' to exit without taking a photo.")
    
    # Try capturing relying on Mac's native AVFoundation backend
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    # Fallback to index 1 if an iPhone/virtual camera is hijacking index 0
    if not cap.isOpened():
        print("Camera at index 0 failed, trying index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    
    if not cap.isOpened():
        print("\n[!] Error: macOS blocked Camera access (or no webcam found).")
        print("Falling back to manual file selection...\n")
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw() # Hide the main tk window
            # Always ensure the popup shows above terminal on Mac
            root.call('wm', 'attributes', '.', '-topmost', True)
            file_path = filedialog.askopenfilename(
                title="Select a Plant Leaf Image to Scan",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
            )
            if file_path:
                print(f"Loaded substitute image: {file_path}")
                img = cv2.imread(file_path)
                return img
            else:
                print("No file selected. Exiting.")
                return None
        except Exception as e:
            print("Error: Could not open file selector.")
            return None
        
    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame")
            break
            
        cv2.imshow("Live Plant Scanner - Press SPACE to capture!", frame)
        
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Scan cancelled.")
            break
        elif k % 256 == 32:
            # SPACE pressed
            print("Snap! Picture taken. Running AI analysis...")
            captured_frame = frame.copy()
            break
            
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    return captured_frame

def segment_leaf(img_rgb):
    """Isolate the leaf target natively using HSV thresholdings."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    segmented_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    return segmented_img, mask

def run_pipeline():
    model, labels = load_labels_and_model()
    
    # 1. Grab image from OpenCV direct webcam GUI
    frame = display_webcam()
    if frame is None:
        return
        
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1b. Validate Leaf Detection!
    segmented_img, mask = segment_leaf(img_rgb)
    leaf_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    leaf_ratio = leaf_pixels / total_pixels
    
    if leaf_ratio < 0.01:  # If less than 1% of the image is "green" leaf material
        print("\n" + "="*50)
        print("❌ REJECTED: NO LEAF DETECTED!")
        print(f"You didn't show a leaf. (Green material recognized: {leaf_ratio*100:.2f}% of area)")
        print("="*50 + "\n")
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title('Captured Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask (Empty)')
        plt.axis('off')
        plt.show()
        return
    
    # 2. Resize and Normalise
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # 3. Model Prediction
    predictions = model.predict(img_batch, verbose=0)
    top_pred_index = np.argmax(predictions[0])
    confidence = predictions[0][top_pred_index] * 100
    predicted_label = labels[top_pred_index]
    
    is_healthy = 'healthy' in predicted_label.lower()
    details = DISEASE_DETAILS.get(predicted_label, "No specific details logged for this variant.")
    
    # 4. Output results inline CLI
    print("\n" + "="*50)
    if is_healthy:
        print(f"✅ VERDICT: HEALTHY (Confidence: {confidence:.2f}%)")
    else:
        print(f"⚠️ VERDICT: DISEASE DETECTED (Confidence: {confidence:.2f}%)")
        
    print(f"🔍 IDENTIFICATION: {predicted_label.replace('_', ' ')}")
    print(f"📝 TREATMENT DETAILS: {details}")
    print("="*50 + "\n")
    
    # 5. Output Graphics Popup
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Live Captured Leaf')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('AI Segmentation Feature Map')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(segmented_img)
    plt.title(f'Extraction Output\nPrediction: {predicted_label}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('camera_prediction.png')
    print("Opening Graphics... (A copy has been saved as 'camera_prediction.png')")
    plt.show()

if __name__ == '__main__':
    run_pipeline()
