import cv2
import numpy as np
import os

# Load Haar Cascade for Face Detection
face_classifier = cv2.CascadeClassifier(
    r"C:/Users/nafee/PycharmProjects/Automatic-Attendance-System/haarcascade_frontalface_default.xml"
)

# Function to Extract Face from Frame
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    return img[y:y + h, x:x + w]

# Ensure dataset directory exists
dataset_path = "C:/Users/nafee/PycharmProjects/Automatic-Attendance-System/dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Ask for user name before starting
user_name = input("Enter the person's name: ").strip()

# Start Video Capture
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()

    face = face_extractor(frame)
    if face is not None:
        count += 1
        face = cv2.resize(face, (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save images with a unique filename
        file_name_path = os.path.join(dataset_path, f'{user_name}_{count}.jpg')
        cv2.imwrite(file_name_path, face)

        # Show the captured face
        cv2.putText(face, f"{user_name} {count}", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    # Exit Condition: Press 'Enter' or collect 100 samples
    if cv2.waitKey(1) == 13 or count == 100:
        break

# Release Camera & Close Windows
cap.release()
cv2.destroyAllWindows()
print(f'✅ Sample Collection Completed for {user_name}!')
