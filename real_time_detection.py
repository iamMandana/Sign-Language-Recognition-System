import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

'''
This file needs to be run in virtual envirnment with necessary libraries installed.
'''

# Config
MODEL_TYPE = "mobilenet"                 # change to "cnn" or "mobilenet" for using different model
MODEL_PATH = "mobilenet_model.keras"     # or mobilenet_model.keras
CONF_THRESH = 0.45

CNN_SIZE = 28
MOBILENET_SIZE = 96
PADDING = 30

#Load models and labels
model = load_model(MODEL_PATH, compile=False)

class_names = np.load("class_names.npy", allow_pickle=True)
letters = {i: class_names[i] for i in range(len(class_names))}

print(f"Loaded model: {MODEL_TYPE}")
print("Classes:", letters)

#Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

#prprosses functions
def preprocess(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if MODEL_TYPE == "cnn":
        gray = cv2.resize(gray, (CNN_SIZE, CNN_SIZE))
        gray = gray.astype(np.float32) / 255.0
        return gray.reshape(1, CNN_SIZE, CNN_SIZE, 1)

    else:  # MobileNet
        gray = cv2.resize(gray, (MOBILENET_SIZE, MOBILENET_SIZE))
        gray = np.stack([gray]*3, axis=-1)   # grayscale → RGB
        gray = gray.astype(np.float32)
        gray = preprocess_input(gray)
        return gray.reshape(1, MOBILENET_SIZE, MOBILENET_SIZE, 3)

#main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        xs = [int(p.x * w) for p in hand.landmark]
        ys = [int(p.y * h) for p in hand.landmark]

        x1 = max(min(xs) - PADDING, 0)
        y1 = max(min(ys) - PADDING, 0)
        x2 = min(max(xs) + PADDING, w)
        y2 = min(max(ys) + PADDING, h)

        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            x = preprocess(crop)

            preds = model.predict(x, verbose=0)[0]
            idx = np.argmax(preds)
            conf = preds[idx]

            if conf > CONF_THRESH:
                label = letters[idx]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,0),
                    2
                )

            # Debug: show model input
            dbg = x[0]
            if MODEL_TYPE == "cnn":
                cv2.imshow("Model Input", dbg.squeeze())
            else:
                dbg = ((dbg + 1) * 127.5).astype(np.uint8)
                cv2.imshow("Model Input", cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
