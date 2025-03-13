import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Set paths for training and testing data
train_data_dir = 'F:\\C_drive_Documents\\Emotion_detection\\Dataset\\train'
test_data_dir = 'F:\\C_drive_Documents\\Emotion_detection\\Dataset\\test'

# Function to count files in each directory
def count_files(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])
    return class_counts

# Count files in training and testing directories
train_counts = count_files(train_data_dir)
test_counts = count_files(test_data_dir)

# Print class distributions
print("Training Set Distribution:")
for class_name, count in train_counts.items():
    print(f"{class_name}: {count}")

print("\nTesting Set Distribution:")
for class_name, count in test_counts.items():
    print(f"{class_name}: {count}")

# Visualize class distribution
def visualize_distribution(counts, title):
    classes = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(10, 5))
    plt.bar(classes, values, color='skyblue')
    plt.xlabel('Emotion Class')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Visualize training and testing distributions
visualize_distribution(train_counts, 'Training Set Distribution')
visualize_distribution(test_counts, 'Testing Set Distribution')

# Display sample images from each class
def display_sample_images(directory, num_images=3):
    classes = os.listdir(directory)
    fig, axes = plt.subplots(len(classes), num_images, figsize=(12, 8))
    fig.tight_layout()

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        for j in range(min(num_images, len(image_files))):
            img_path = os.path.join(class_dir, image_files[j])
            try:
                img = Image.open(img_path)
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].set_title(class_name)
                axes[i, j].axis('off')
            except FileNotFoundError:
                print(f"File not found: {img_path}")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    plt.show()

# Display sample images from the training set
print("\nSample Images from Training Set:")
display_sample_images(train_data_dir)
