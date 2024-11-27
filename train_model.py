import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from skimage.transform import resize  # To resize images if needed

# Load the preprocessed data
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Optionally, downsample images (e.g., reduce resolution to save memory)
# If the resolution is too high, this can help with memory constraints
X_train = np.array([resize(img, (64, 64)) for img in X_train])
X_val = np.array([resize(img, (64, 64)) for img in X_val])

# Model definition
model = Sequential([
    Input(shape=(64, 64, 3)),  # Adjusted input layer for downsampled images
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Adding dropout to reduce overfitting
    Dense(y.shape[1], activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the data augmentation pipeline
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data augmentation on the training set
# Since we don't need to call fit() for ImageDataGenerator, we can skip it
# datagen.fit(X_train)  # This line is removed

# Define EarlyStopping and ModelCheckpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),  # Reduced batch size
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint]
)

# Save the trained model
model.save("final_model.keras")

print("Model training complete!")
