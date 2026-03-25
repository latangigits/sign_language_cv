import cv2
import os

label = "hello"   # change this every time

save_path = f"dataset/{label}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

print("Press SPACE to capture | Q to quit")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    x1, y1 = w//4, h//4
    x2, y2 = 3*w//4, 3*h//4

    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # SPACE
        img = cv2.resize(roi, (48,48))
        cv2.imwrite(f"{save_path}/{count}.jpg", img)
        print("Saved:", count)
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()