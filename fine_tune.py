import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import os
from PIL import Image

# Set paths for training and testing data
train_data_dir = 'F:\\C_drive_Documents\\Emotion_detection\\Dataset\\train'
test_data_dir = 'F:\\C_drive_Documents\\Emotion_detection\\Dataset\\test'

# Image dimensions
img_width, img_height = 48, 48
num_classes = 7

# Batch size
batch_size = 64

# Number of epochs
epochs = 60


# Load and preprocess data
def load_data(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = Image.open(img_path).convert('L').resize((img_width, img_height))
                    img_array = np.array(img) / 255.0  # Normalize pixel values
                    images.append(img_array)
                    labels.append(class_names.index(class_name))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels), class_names


# Load training data
X_train, y_train, class_names = load_data(train_data_dir)

# Reshape for SMOTE
X_train = X_train.reshape(-1, img_width * img_height)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Reshape back
X_train_resampled = X_train_resampled.reshape(-1, img_width, img_height, 1)

# Convert labels to categorical format
y_train_resampled = tf.keras.utils.to_categorical(y_train_resampled, num_classes=num_classes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)


# CNN Model Architecture (Based on Video)
def build_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    inputs = tf.keras.Input(shape=input_shape)

    # Conv Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Conv Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Conv Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Flatten and Dense Layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Load the saved model
model = load_model('emotion_detection_model_smote_cnn_video.keras')

# Fine-tune the model
optimizer = Adam(learning_rate=0.00001)  # Lower learning rate
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=(X_val, y_val),
    epochs=epochs,
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)

# Save the fine-tuned model
model.save('emotion_detection_model_fine_tuned.keras')
print("Model saved as 'emotion_detection_model_fine_tuned.keras'")
