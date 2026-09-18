import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

labels = ["A","B","C","D","E","G","H","I","J","K","L","M","N","O","P","S","T","U","V","SPACE","DELETE","R","W","X","Y","Z"]

X = []
y = []

for label in labels:

    folder = f"dataset/{label}"

    for file in os.listdir(folder):

        if file.endswith(".npy"):

            feature = np.load(os.path.join(folder, file))

            X.append(feature)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("Y shape:", y.shape)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(labels, "labels.pkl")

print("Training Complete!")