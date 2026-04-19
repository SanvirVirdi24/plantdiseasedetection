import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import json

# Set Streamlit page configuration
st.set_page_config(page_title="Plant Disease Detector", layout="wide", page_icon="🌿")

# Load details dictionary to give insights on predicted diseases
DISEASE_DETAILS = {
    'Pepper__bell___Bacterial_spot': "A bacterial disease that causes water-soaked spots on leaves and fruit. **Treatment:** Use copper-based bactericides.",
    'Pepper__bell___healthy': "No diseases detected! The plant is healthy. Keep up the good work.",
    'Potato___Early_blight': "A fungal disease causing dark lesions with concentric rings on older leaves. **Treatment:** Needs proper fungicide and crop rotation.",
    'Potato___Late_blight': "A highly destructive fungal disease causing rapid decay of leaves and tubers. **Treatment:** Apply specific fungicides, destroy severely affected plants.",
    'Potato___healthy': "No diseases detected! The plant is healthy. Keep up the good work.",
    'Tomato_Bacterial_spot': "Small water-soaked spots on leaves. Can cause severe defoliation. **Treatment:** Practice crop rotation and use copper sprays.",
    'Tomato_Early_blight': "Causes 'bullseye' spots on lower leaves, eventually turning leaves yellow. **Treatment:** Treat with protective fungicides and avoid overhead watering.",
    'Tomato_Late_blight': "Causes grey to brown lesions on leaves and stems. Extremely contagious. **Treatment:** Destroy infected plants and apply targeted fungicides.",
    'Tomato_Leaf_Mold': "Causes pale green to yellow spots on the upper leaf surface and velvety mold on the bottom. **Treatment:** Improve air circulation and reduce humidity.",
    'Tomato_Septoria_leaf_spot': "Fungal disease causing small circular spots with grey centers. **Treatment:** Remove infected leaves, mulch, and use fungicide.",
    'Tomato_Spider_mites_Two_spotted_spider_mite': "Tiny pests that suck sap, causing yellow stippling and visible webbing. **Treatment:** Use miticides, insecticidal soap, or neem oil.",
    'Tomato__Target_Spot': "Fungal pathogen causing brown spots on leaves and fruit. **Treatment:** Ensure proper spacing and apply appropriate fungicides.",
    'Tomato__Tomato_YellowLeaf__Curl_Virus': "Viral disease transmitted by whiteflies. Causes yellowing and curving of leaves. **Treatment:** Control the whitefly population and use resistant varieties.",
    'Tomato__Tomato_mosaic_virus': "Viral disease causing mottled, mosaic-like patterns on leaves. Highly transmissible. **Treatment:** Destroy infected plants immediately; wash hands/tools.",
    'Tomato_healthy': "No diseases detected! The plant is healthy. Keep up the good work."
}

IMG_HEIGHT = 128
IMG_WIDTH = 128

@st.cache_resource
def load_model_and_labels():
    model_path = 'models/plant_disease_model.h5'
    labels_path = 'models/class_indices.json'
    
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None, None
        
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, 'r') as f:
        class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

def segment_leaf(img_rgb):
    """HSV color thresholding to segment out green leaf regions"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    segmented_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    return segmented_img, mask

def predict_image(image_file, model, labels):
    # Convert uploaded file/camera image to an OpenCV compatible numpy array
    image = Image.open(image_file).convert('RGB')
    img_rgb = np.array(image)
    
    # Generate visual segmentation for display
    segmented_img, mask = segment_leaf(img_rgb)
    
    # Validate Leaf Detection
    leaf_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    leaf_ratio = leaf_pixels / total_pixels
    
    if leaf_ratio < 0.01:
        return "NO_LEAF", 0.0, img_rgb, mask, segmented_img
    
    # Preprocess
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    predictions = model.predict(img_batch)
    top_pred_index = np.argmax(predictions[0])
    confidence = predictions[0][top_pred_index] * 100
    predicted_label = labels[top_pred_index]
    
    return predicted_label, confidence, img_rgb, mask, segmented_img

# ── Per-class stats from model_stats.txt ──────────────────────────────────────
CLASS_STATS = {
    'Pepper bell\nBacterial spot':          {'precision': 0.91, 'recall': 0.81, 'f1': 0.85, 'support': 200},
    'Pepper bell\nhealthy':                 {'precision': 0.94, 'recall': 0.95, 'f1': 0.94, 'support': 296},
    'Potato\nEarly blight':                 {'precision': 0.98, 'recall': 0.90, 'f1': 0.93, 'support': 200},
    'Potato\nLate blight':                  {'precision': 0.72, 'recall': 0.94, 'f1': 0.82, 'support': 200},
    'Potato\nhealthy':                      {'precision': 0.84, 'recall': 0.68, 'f1': 0.75, 'support':  31},
    'Tomato\nBacterial spot':               {'precision': 0.93, 'recall': 0.96, 'f1': 0.95, 'support': 426},
    'Tomato\nEarly blight':                 {'precision': 0.73, 'recall': 0.75, 'f1': 0.74, 'support': 200},
    'Tomato\nLate blight':                  {'precision': 0.89, 'recall': 0.81, 'f1': 0.85, 'support': 382},
    'Tomato\nLeaf Mold':                    {'precision': 0.91, 'recall': 0.87, 'f1': 0.89, 'support': 191},
    'Tomato\nSeptoria leaf spot':           {'precision': 0.89, 'recall': 0.83, 'f1': 0.86, 'support': 355},
    'Tomato\nSpider mites':                 {'precision': 0.95, 'recall': 0.88, 'f1': 0.91, 'support': 336},
    'Tomato\nTarget Spot':                  {'precision': 0.80, 'recall': 0.94, 'f1': 0.87, 'support': 281},
    'Tomato\nYellowLeaf Curl Virus':        {'precision': 0.97, 'recall': 0.98, 'f1': 0.97, 'support': 642},
    'Tomato\nmosaic virus':                 {'precision': 0.91, 'recall': 0.96, 'f1': 0.94, 'support':  75},
    'Tomato\nhealthy':                      {'precision': 1.00, 'recall': 0.98, 'f1': 0.99, 'support': 319},
}

# UI Layout
st.title("🌿 AI Plant Disease Detector App")
st.markdown("""
Welcome to the Plant Disease Classification Dashboard! 
You can **click a live picture** of a leaf using your camera or **upload** an existing image below. 
The AI will scan the leaf, detect any diseases, and provide you with detailed information/treatments!
""")

model, labels = load_model_and_labels()

if model is None:
    st.error("🚨 **Model Not Found!** 🚨 \nIt looks like the model is still training or hasn't been built yet. Please wait for `train.py` to finish saving `models/plant_disease_model.h5`.")
    st.stop()

# Interactive Taps for Selection
tab1, tab2, tab3 = st.tabs(["📸 Click a live picture", "📂 Upload a file", "📊 Model Analytics"])

with tab1:
    st.write("Use your computer or phone webcam to snap a photo.")
    camera_image = st.camera_input("Take a picture of the plant leaf")
    
with tab2:
    st.write("Upload an image file directly from your computer.")
    uploaded_image = st.file_uploader("Choose a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

with tab3:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    st.header("📊 Model Performance Analytics")
    st.markdown("Full breakdown of how the AI model performs across all **15 disease classes** on the held-out test set.")

    # ── Overall KPI row ───────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🎯 Overall Accuracy", "90.32%")
    k2.metric("📐 Macro Avg F1", "0.88")
    k3.metric("⚖️ Weighted Avg F1", "0.90")
    k4.metric("🗂️ Test Samples", "4,134")

    st.markdown("---")

    # ── Weakest classes callout ────────────────────────────────────────────────
    weak = {k: v for k, v in CLASS_STATS.items() if v['f1'] < 0.85}
    if weak:
        st.warning(
            "⚠️ **Lowest-performing classes** (F1 < 0.85) — these are where the model struggles most:\n\n" +
            "\n".join([
                f"- **{k.replace(chr(10), ' ')}** → F1: {v['f1']:.2f}  |  Precision: {v['precision']:.2f}  |  Recall: {v['recall']:.2f}"
                for k, v in sorted(weak.items(), key=lambda x: x[1]['f1'])
            ])
        )

    # ── F1 horizontal bar chart ────────────────────────────────────────────────
    st.subheader("Per-Class F1 Score")
    st.caption("F1 score combines precision and recall — a score of 1.0 is perfect. Dashed line = 90% threshold.")

    classes = list(CLASS_STATS.keys())
    f1_scores    = [CLASS_STATS[c]['f1']        for c in classes]
    colors = ['#e74c3c' if s < 0.80 else '#f39c12' if s < 0.90 else '#2ecc71' for s in f1_scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#161b22')

    bars = ax.barh(classes[::-1], f1_scores[::-1], color=colors[::-1], height=0.55, edgecolor='none')

    for bar, val in zip(bars, f1_scores[::-1]):
        ax.text(bar.get_width() + 0.007, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', ha='left', color='white', fontsize=9, fontweight='bold')

    ax.set_xlim(0, 1.12)
    ax.set_xlabel('F1 Score', color='#aaaaaa', fontsize=10)
    ax.tick_params(colors='white', labelsize=9)
    ax.spines[:].set_visible(False)
    ax.axvline(0.90, color='white', linestyle='--', linewidth=0.8, alpha=0.35)
    ax.set_facecolor('#161b22')

    legend_patches = [
        mpatches.Patch(color='#e74c3c', label='< 0.80  Weak'),
        mpatches.Patch(color='#f39c12', label='0.80 – 0.89  Moderate'),
        mpatches.Patch(color='#2ecc71', label='≥ 0.90  Strong'),
    ]
    ax.legend(handles=legend_patches, loc='lower right',
              facecolor='#1e2130', edgecolor='none', labelcolor='white', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Precision vs Recall vs F1 grouped bar chart ───────────────────────────
    st.subheader("Precision vs Recall vs F1 — Side by Side")
    st.caption("Abbreviated class names on X-axis. Hover for full labels.")

    x = list(range(len(classes)))
    width = 0.26
    precision_vals = [CLASS_STATS[c]['precision'] for c in classes]
    recall_vals    = [CLASS_STATS[c]['recall']    for c in classes]
    f1_vals        = [CLASS_STATS[c]['f1']        for c in classes]
    short_labels   = [c.replace('\n', ' ') for c in classes]

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    fig2.patch.set_facecolor('#0e1117')
    ax2.set_facecolor('#161b22')

    ax2.bar([i - width for i in x], precision_vals, width, label='Precision', color='#3498db', alpha=0.9, edgecolor='none')
    ax2.bar([i          for i in x], recall_vals,    width, label='Recall',    color='#9b59b6', alpha=0.9, edgecolor='none')
    ax2.bar([i + width  for i in x], f1_vals,        width, label='F1 Score',  color='#1abc9c', alpha=0.9, edgecolor='none')

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, rotation=40, ha='right', color='white', fontsize=8)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Score', color='#aaaaaa')
    ax2.tick_params(colors='white')
    ax2.spines[:].set_visible(False)
    ax2.legend(facecolor='#1e2130', edgecolor='none', labelcolor='white', fontsize=10)
    ax2.axhline(1.0, color='white', linewidth=0.5, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")

    # ── Training graphs + confusion matrix side by side ───────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📈 Training History")
        st.caption("Accuracy and loss over 10 training epochs.")
        if os.path.exists('training_graphs.png'):
            st.image('training_graphs.png', use_container_width=True)
        else:
            st.info("training_graphs.png not found.")

    with col_b:
        st.subheader("🔲 Confusion Matrix")
        st.caption("Rows = actual, columns = predicted. Bright diagonal = correct predictions.")
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', use_container_width=True)
        else:
            st.info("confusion_matrix.png not found.")

    st.markdown("---")

    # ── Support (sample count) bar chart ──────────────────────────────────────
    st.subheader("Test Set Size per Class")
    st.caption("⚠️ Classes with very few samples (e.g. Potato healthy = 31) tend to have lower and less reliable F1 scores.")

    support_vals = [CLASS_STATS[c]['support'] for c in classes]
    sup_colors   = ['#e74c3c' if s < 100 else '#f39c12' if s < 250 else '#3498db' for s in support_vals]

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    fig3.patch.set_facecolor('#0e1117')
    ax3.set_facecolor('#161b22')
    ax3.bar(short_labels, support_vals, color=sup_colors, edgecolor='none', width=0.6)
    ax3.set_xticks(range(len(short_labels)))
    ax3.set_xticklabels(short_labels, rotation=40, ha='right', color='white', fontsize=8)
    ax3.set_ylabel('# test images', color='#aaaaaa')
    ax3.tick_params(colors='white')
    ax3.spines[:].set_visible(False)
    for i, v in enumerate(support_vals):
        ax3.text(i, v + 8, str(v), ha='center', color='white', fontsize=8, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown("---")
    st.info("""
**Why might accuracy differ on real photos?**
- The model was trained on controlled lab images — real-world lighting, angles, and backgrounds can reduce confidence.
- Classes like *Tomato Early blight* (F1: 0.74) and *Potato healthy* (F1: 0.75) are genuinely harder due to visual similarity to other classes or limited training data.
- To improve accuracy: collect more diverse real-world samples and retrain with data augmentation.
""")

# Logic Selection
selected_target = camera_image if camera_image is not None else uploaded_image

if selected_target is not None:
    st.markdown("---")
    with st.spinner("Analyzing Leaf with AI Model..."):
        try:
            pred_label, conf, orig_img, mask, seg_img = predict_image(selected_target, model, labels)
            
            if pred_label == "NO_LEAF":
                st.header("Results: ❌ Rejected")
                st.error("⚠️ **No Leaf Detected!** \nThe AI couldn't find enough plant material (greenery) in the photo. Please make sure the leaf is clearly visible.")
            else:
                # Identify if it's healthy
                is_healthy = 'healthy' in pred_label.lower()
                
                # Status Layout
                st.header(f"Results: {pred_label.replace('_', ' ')}")
                
                # Confidence Metrics Wrapper
                if is_healthy:
                    st.success(f"✅ **Status:** Health completely verified! (Confidence: {conf:.2f}%)")
                else:
                    st.warning(f"⚠️ **Status:** Disease Detected! (Confidence: {conf:.2f}%)")
                
                st.subheader("Disease Details & Treatment")
                details = DISEASE_DETAILS.get(pred_label, "No specific details available for this label.")
                st.info(details)
            
            # Visualization Images
            st.subheader("Visual Analysis Breakdown")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(orig_img, caption="Original Image", use_container_width=True)
            with col2:
                # Mask needs to be outputting properly as gray mapping
                st.image(mask, caption="AI Feature Mask (Green Extraction)", use_container_width=True, clamp=True)
            with col3:
                st.image(seg_img, caption="Segmented Target Area", use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            
st.markdown("---")
st.caption("Final Year Project - AI Driven Agronomy Analytics")
