import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

# MediaPipe setup
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    prediction = "No Hand"

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

            mpDraw.draw_landmarks(
                frame,
                hand_landmarks,
                mpHands.HAND_CONNECTIONS
            )

        # 126 features
        final_feature = left_features + right_features

        # Debug (remove later)
        # print(len(final_feature))

        prediction = model.predict([final_feature])[0]

    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Indian Sign Language Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()