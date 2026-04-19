import os
import shutil
import random

def split_dataset(source_dir, dest_dir, split_ratio=0.8):
    '''
    Splits the dataset into training and testing sets.
    '''
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    
    # Create the output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Ignore these folders if present in the source dir
    ignore_folders = ['dataset', 'PlantVillage', '.DS_Store', '__pycache__']
    
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and d not in ignore_folders]
    
    print(f"Found {len(classes)} classes.")
    
    for cls in classes:
        cls_source = os.path.join(source_dir, cls)
        cls_train = os.path.join(train_dir, cls)
        cls_test = os.path.join(test_dir, cls)
        
        os.makedirs(cls_train, exist_ok=True)
        os.makedirs(cls_test, exist_ok=True)
        
        images = [f for f in os.listdir(cls_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        print(f"Class '{cls}': {len(train_images)} train, {len(test_images)} test")
        
        for img in train_images:
            shutil.copy(os.path.join(cls_source, img), os.path.join(cls_train, img))
            
        for img in test_images:
            shutil.copy(os.path.join(cls_source, img), os.path.join(cls_test, img))

if __name__ == '__main__':
    # Using the inner PlantVillage directory as it seems to be duplicated
    source_directory = './PlantVillage'
    destination_directory = './dataset'
    
    print("Starting dataset split...")
    split_dataset(source_directory, destination_directory)
    print("Dataset split complete. Located in ./dataset")
