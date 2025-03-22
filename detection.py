import cv2
import numpy as np
import os
import pandas as pd
import pickle
from datetime import datetime

# Load Trained LBPH Model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("C:/Users/nafee/PycharmProjects/Automatic-Attendance-System/trained_model.xml")

# Load Haar Cascade for Face Detection
face_classifier = cv2.CascadeClassifier(
    r"C:/Users/nafee/PycharmProjects/Automatic-Attendance-System/haarcascade_frontalface_default.xml"
)

# Load Label Mapping
with open("C:/Users/nafee/PycharmProjects/Automatic-Attendance-System/labels.pkl", "rb") as f:
    label_dict = pickle.load(f)  # {name: label}
label_to_name = {v: k for k, v in label_dict.items()}  # Reverse mapping

# Create CSV Attendance File
today_date = datetime.now().strftime("%Y-%m-%d")
attendance_file = f"C:/Users/nafee/PycharmProjects/Automatic-Attendance-System/Attendance_{today_date}.csv"

if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Status", "Time"])
    df.to_csv(attendance_file, index=False)

# Function to Detect Faces
def preprocess_image(image):
    """Enhance image contrast and remove noise."""
    image = cv2.equalizeHist(image)  # Improves contrast
    image = cv2.GaussianBlur(image, (3, 3), 0)  # Reduces noise
    return image

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img, None, []

    face_list = []
    locations = []
    for (x, y, w, h) in faces:
        face_list.append(preprocess_image(gray[y:y + h, x:x + w]))  # Apply preprocessing
        locations.append((x, y, w, h))

    return img, face_list, locations

# Open Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Camera not detected!")
        break

    image, faces, locations = face_detector(frame)

    if faces is not None:
        for i, face in enumerate(faces):
            face_resized = cv2.resize(face, (200, 200))
            label, confidence = model.predict(face_resized)
            confidence_score = int(100 * (1 - confidence / 300))

            if label in label_to_name and confidence_score > 80:  # Increased confidence threshold
                recognized_name = label_to_name[label]
                color = (0, 255, 0)  # Green for recognized
            else:
                recognized_name = "Unknown"
                color = (0, 0, 255)  # Red for unknown

            x, y, w, h = locations[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

            # Mark Attendance
            df = pd.read_csv(attendance_file)
            if recognized_name != "Unknown" and recognized_name not in df["Name"].values:
                new_entry = pd.DataFrame([[recognized_name, "Present", datetime.now().strftime("%H:%M:%S")]],
                                         columns=["Name", "Status", "Time"])
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_csv(attendance_file, index=False)

    cv2.imshow("Face Recognition - Attendance System", image)

    if cv2.waitKey(1) == 13:  # Press Enter to exit
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Attendance Marked Successfully!")

# Mark Absent
df = pd.read_csv(attendance_file)
for user in label_dict.keys():
    if user not in df["Name"].values:
        new_entry = pd.DataFrame([[user, "Absent", "-"]], columns=["Name", "Status", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)

df.to_csv(attendance_file, index=False)
print("✅ Final Attendance List Updated!")
