Automatic Attendance System using OpenCV & Face Recognition 🎯
✨ Features:
✅ Real-time Face Detection using OpenCV Haar Cascades
✅ Automatic Attendance Logging with timestamps
✅ CCTV Integration for continuous monitoring
✅ Dataset Creation & Training for multiple users
✅ Secure & Scalable with database storage

🔧 Technologies Used:
Python (OpenCV, NumPy, Pandas)
Haar Cascades for Face Detection
LBPH Face Recognizer for Face Recognition
SQLite/MySQL for attendance storage
Flask/Django (Optional) for web-based monitoring
📸 How It Works:
1️⃣ The system captures faces from a CCTV feed.
2️⃣ The Haar Cascade Classifier detects faces.
3️⃣ The LBPH Face Recognizer identifies individuals.
4️⃣ Attendance is automatically marked in the database.
5️⃣ Admins can view and export attendance reports.

📂 Project Structure:
bash
Copy
Edit
├── dataset/                 # Collected face images  
├── models/                  # Trained face recognition model  
├── attendance.db            # SQLite/MySQL Database  
├── training.py              # Model training script  
├── dataset.py               # Face data collection  
├── recognize.py             # Main face recognition & attendance system  
└── README.md                # Project documentation  
🚀 Setup & Installation:
bash
Copy
Edit
git clone https://github.com/yourusername/Automatic-Attendance-System.git
cd Automatic-Attendance-System
pip install -r requirements.txt
python dataset.py   # Collect face images
python training.py  # Train the model
python recognize.py # Start the attendance system
📌 Future Improvements:
✅ Cloud Integration for remote access
✅ Mobile App for attendance reports
✅ Multi-camera Support
