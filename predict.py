import cv2
import numpy as np
import time
import pyttsx3
import threading
from tensorflow.keras.models import load_model

# ------------------ SPEECH SETUP ------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    threading.Thread(target=_speak, args=(text,)).start()

def _speak(text):
    engine.say(text)
    engine.runAndWait()

# ------------------ LOAD MODEL ------------------
model = load_model("gesture_model.h5")

labels = ["come", "hello", "no", "stop", "yes"]

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0)

print("Press Q to quit")

# ------------------ SPEECH CONTROL ------------------
last_spoken = ""
last_time = 0
cooldown = 2   # seconds

# ------------------ MAIN LOOP ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # ROI box
    x1, y1 = w//4, h//4
    x2, y2 = 3*w//4, 3*h//4

    roi = frame[y1:y2, x1:x2]

    # Preprocess
    img = cv2.resize(roi, (48, 48))
    img = img / 255.0
    img = np.reshape(img, (1, 48, 48, 3))

    # Prediction
    pred = model.predict(img, verbose=0)[0]
    idx = np.argmax(pred)
    confidence = pred[idx]

    label = "..."

    # ------------------ PREDICTION + SPEECH ------------------
    if confidence > 0.85:
        label = labels[idx]

        current_time = time.time()

        if label != last_spoken and (current_time - last_time > cooldown):
            speak(label)
            last_spoken = label
            last_time = current_time

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Show label
    cv2.putText(frame, f"{label} ({confidence:.2f})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    # Display
    cv2.imshow("Gesture Detection", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------ CLEANUP ------------------
cap.release()
cv2.destroyAllWindows()