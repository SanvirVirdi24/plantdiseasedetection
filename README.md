Plant Disease Classification Project
A lightweight computer vision workflow using Python for identifying and classifying plant leaf diseases.

Key Features
Automatic Dataset Distribution: Splits the raw data automatically into training (80%) and testing (20%) subgroups.
Basic Image Preprocessing: Properly normalizes raw images arrays via scaling.
Simplified CNN Modeling: Constructs a simple 3-layer Convolutional Neural Network (Conv2D -> ReLU -> MaxPooling2D -> Dense) with Dropout. Keeps code simple but highly effective for general classification.
Leaf Color Segmentation (OpenCV): Features simple leaf segmentation using HSV color thresholding masks to extract boundaries on plant testing.
Performance Evaluations: Tracks epoch history and visually generates Accuracy/Loss graphs and a complete categorical Confusion Matrix.
Random Localized Predictor: Pass un-trained images into the pipeline or let it pick random samples and visualize confidence maps.
Project Structure Overview
.
├── dataset/                    # Generated folder holding separated 'train' and 'test' images
├── split_data.py               # Utility script to construct the dataset layout
├── train.py                    # Core pipeline to build, train, evaluate, and save CNN model
├── predict.py                  # Utility script to process segmentations and display predictions
├── models/                     # Output repository where the network state (h5) and configurations save
├── prediction_output.png       # Generated visualization grid
├── training_graphs.png         # Generated metrics for model evaluation
└── confusion_matrix.png        # Generated heatmap
How to use
1. Data Set Setup (Already Completed!)
The split_data.py setup logic has already been executed! Raw images have safely been organized and populated into dataset/train and dataset/test.

2. Train the CNN Model
Run the primary training script via terminal. Wait a few moments as your computer iterates over epochs.

python3 train.py
This step outputs training_graphs.png, confusion_matrix.png, and produces models/plant_disease_model.h5 artifacts automatically.

3. Run Inference and Mask Leaf Outcomes
To test out the deployed model logic, execute predict.py. Have a specific leaf image to inspect? Pass it logically as an argument!

# Run against a random testing image
python3 predict.py 

# Or test a specific customized photo!
python3 predict.py dataset/test/Potato___Early_blight/example_image.jpg
Creates the CLI output text as well drawing the visualization output to prediction_output.png.
