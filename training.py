import cv2
import numpy as np
import os
import pickle

# Path to dataset
dataset_path = "C:/Users/nafee/PycharmProjects/Automatic-Attendance-System/dataset"

# Load Haar Cascade for Face Detection
face_classifier = cv2.CascadeClassifier(
    r"C:/Users/nafee/PycharmProjects/Automatic-Attendance-System/haarcascade_frontalface_default.xml"
)

# Prepare training data
faces = []
labels = []
label_dict = {}
current_id = 0

for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):
        name = filename.split("_")[0]  # Extract name from filename

        if name not in label_dict:
            label_dict[name] = current_id
            current_id += 1

        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (200, 200))

        faces.append(img_resized)
        labels.append(label_dict[name])

# Convert to numpy arrays
faces = np.array(faces, dtype="uint8")
labels = np.array(labels, dtype="int")

# Train the LBPH Face Recognizer
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

# Save trained model
model.save("C:/Users/nafee/PycharmProjects/Automatic-Attendance-System/trained_model.xml")

# Save label dictionary
with open("C:/Users/nafee/PycharmProjects/Automatic-Attendance-System/labels.pkl", "wb") as f:
    pickle.dump(label_dict, f)

print("✅ Model Trained and Saved Successfully!")
