import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
from sklearn.utils import shuffle

# IMPORTANT: fixed labels
labels = ["come", "hello", "no", "stop", "yes"]

data = []
target = []
img_size = 48

# load images
for label in labels:
    path = os.path.join("dataset", label)
    class_num = labels.index(label)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        img_array = cv2.imread(img_path)
        if img_array is None:
            continue

        img_array = cv2.resize(img_array, (img_size, img_size))

        data.append(img_array)
        target.append(class_num)

# normalize
data = np.array(data) / 255.0
target = np.array(target)

# shuffle data
data, target = shuffle(data, target)

print("✅ Data loaded")

# CNN model (better)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("🚀 Training started...")
model.fit(data, target, epochs=20)

model.save("gesture_model.h5")

print("✅ Model Trained & Saved!")