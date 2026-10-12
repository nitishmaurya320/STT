from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import joblib
from googletrans import Translator

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",
                "http://localhost:5174",
                "http://127.0.0.1:5173",
                "https://stt-pied-ten.vercel.app"
            ]
        }
    },
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# Load model
model = joblib.load("model.pkl")

print("Model expects:", model.n_features_in_, "features")

# MediaPipe setup
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)


def extract_landmarks(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

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

    final_feature = left_features + right_features

    return final_feature


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "text": "No Image"
        })

    file = request.files["image"]

    img_bytes = np.frombuffer(
        file.read(),
        np.uint8
    )

    image = cv2.imdecode(
        img_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return jsonify({
            "text": "Invalid Image",
            "confidence": 0
        })

    features = extract_landmarks(image)

    if features is None:
        return jsonify({
            "text": "No Hand",
            "confidence": 0
        })

    print("Feature Length:", len(features))

    prediction = model.predict([features])[0]

    confidence = float(
        np.max(
            model.predict_proba([features])[0]
        )
    )

    return jsonify({
        "text": prediction,
        "confidence": round(confidence, 4)
    })

translator = Translator()

@app.route("/translate", methods=["POST"])
def translate_text():
    data = request.get_json()

    text = data.get("text", "")

    if not text:
        return jsonify({"translation": ""})

    result = translator.translate(text, dest="hi")

    return jsonify({
        "translation": result.text
    })
@app.route("/")
def home():
    return "Indian Sign Language API Running"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )