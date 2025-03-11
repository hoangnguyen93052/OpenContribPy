import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (224, 224)
MODEL_PATH = 'object_detection_model.h5'
DATASET_PATH = 'dataset/'
LABELS = ['cat', 'dog']

# Function to load images and labels
def load_data():
    images = []
    labels = []
    for label in LABELS:
        path = os.path.join(DATASET_PATH, label, '*.jpg')
        for img_path in glob.glob(path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img)
            labels.append(LABELS.index(label))
    return np.array(images), np.array(labels)

# Function to preprocess images
def preprocess_images(images):
    images = images / 255.0  # Normalize images
    return images

# Function to create a convolutional neural network model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(LABELS), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(images, labels):
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    model = create_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    model.save(MODEL_PATH)
    return model

# Function to load the trained model
def load_trained_model():
    return load_model(MODEL_PATH)

# Function to predict the class of an image
def predict_image(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    return LABELS[np.argmax(prediction)]

# Function to display predictions
def display_predictions(image_paths, model):
    for path in image_paths:
        prediction = predict_image(path, model)
        img = cv2.imread(path)
        cv2.putText(img, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Prediction', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    images, labels = load_data()
    images = preprocess_images(images)
    train_model(images, labels)
    model = load_trained_model()

    # Test predictions
    test_images = ['test_image_1.jpg', 'test_image_2.jpg']
    display_predictions(test_images, model)
    
    plt.imshow(images[0])
    plt.title(f'Label: {LABELS[labels[0]]}')
    plt.show()