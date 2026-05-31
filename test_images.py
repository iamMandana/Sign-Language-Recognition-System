import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

'''
This file needs to be run in virtual envirnment with necessary libraries installed.
'''

#Config
IMG_SIZE_CNN = 28
IMG_SIZE_MB = 96

N_SAMPLES = 10
DATASET_DIR = "Dataset"

CNN_PATH = "cnn_model.keras"
MB_PATH  = "mobilenet_model.keras"

#Load models
cnn = load_model(CNN_PATH, compile=False)
mobilenet = load_model(MB_PATH, compile=False)

class_names = np.load("class_names.npy", allow_pickle=True)
letters = {i: class_names[i] for i in range(len(class_names))}

print("Loaded labels:", letters)

#Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  
    max_num_hands=1,
    min_detection_confidence=0.6
)

#Hand detection
def detect_and_crop(img):
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0]
    xs = [int(p.x * w) for p in lm.landmark]
    ys = [int(p.y * h) for p in lm.landmark]

    padding = 20
    x1, x2 = max(min(xs) - padding, 0), min(max(xs) + padding, w)
    y1, y2 = max(min(ys) - padding, 0), min(max(ys) + padding, h)

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop

#Preprocessing
#for CNN
def preprocess_cnn(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE_CNN, IMG_SIZE_CNN))
    gray = gray.astype("float32") / 255.0
    return gray.reshape(1, IMG_SIZE_CNN, IMG_SIZE_CNN, 1)

#For mobilenet
def preprocess_mobilenet(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE_MB, IMG_SIZE_MB))
    rgb = rgb.astype("float32")
    rgb = preprocess_input(rgb)   # MobileNet normalization
    return np.expand_dims(rgb, axis=0)

#load images
images = [
    os.path.join(root, f)
    for root, _, files in os.walk(DATASET_DIR)
    for f in files if f.lower().endswith(".jpg")
]

samples = random.sample(images, min(N_SAMPLES, len(images)))

#visual evaluation
plt.figure(figsize=(16, 6))
shown = 0

for path in samples:

    img = cv2.imread(path)
    if img is None:
        continue

    crop = detect_and_crop(img)
    if crop is None:
        continue

    x_cnn = preprocess_cnn(crop)
    x_mb  = preprocess_mobilenet(crop)

    #Predictions
    cnn_probs = cnn.predict(x_cnn, verbose=0)[0]
    mb_probs  = mobilenet.predict(x_mb, verbose=0)[0]

    cnn_idx = np.argmax(cnn_probs)
    mb_idx  = np.argmax(mb_probs)

    #Plot
    plt.subplot(2, 5, shown + 1)

    #Show grayscale CNN input for consistency
    plt.imshow(x_cnn[0].squeeze(), cmap="gray")

    plt.title(
        f"CNN: {letters[cnn_idx]} ({cnn_probs[cnn_idx]:.2f})\n"
        f"MB:  {letters[mb_idx]} ({mb_probs[mb_idx]:.2f})"
    )

    plt.axis("off")

    shown += 1
    if shown == 10:
        break

plt.suptitle("Real-world Gesture Evaluation: CNN vs MobileNet", fontsize=14)
plt.tight_layout()
plt.show()
