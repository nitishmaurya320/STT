import cv2
import mediapipe as mp
import numpy as np
import os
import time

LETTER = "N"
SAMPLES = 1800

START_DELAY = 15      # wait 15 seconds
CAPTURE_TIME = 60     # collect all samples in 60 sec

save_path = f"dataset/{LETTER}"
os.makedirs(save_path, exist_ok=True)

mpHands = mp.solutions.hands

hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(2)

count = 0

interval = CAPTURE_TIME / SAMPLES
last_saved = 0

start_time = time.time()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    elapsed = time.time() - start_time

    # Countdown
    if elapsed < START_DELAY:
        remaining = int(START_DELAY - elapsed)

        cv2.putText(
            frame,
            f"Starting in {remaining}s",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("Collect Data", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        left_features = [0] * 63
        right_features = [0] * 63

        for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness):

            label = handedness.classification[0].label

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            zs = [lm.z for lm in hand_landmarks.landmark]

            minx = min(xs)
            miny = min(ys)
            minz = min(zs)

            feature = []

            for lm in hand_landmarks.landmark:
                feature.extend([
                    lm.x - minx,
                    lm.y - miny,
                    lm.z - minz
                ])

            if label == "Left":
                left_features = feature
            else:
                right_features = feature

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mpHands.HAND_CONNECTIONS
            )

        final_feature = left_features + right_features

        current_time = time.time()

        if current_time - last_saved >= interval:

            np.save(
                os.path.join(save_path, f"{count}.npy"),
                np.array(final_feature)
            )

            count += 1
            last_saved = current_time

            print(f"Saved {count}/{SAMPLES}")

    cv2.putText(
        frame,
        f"{LETTER} {count}/{SAMPLES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collect Data", frame)

    if count >= SAMPLES:
        print("Collection Complete!")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()