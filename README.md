🚗 Driver Drowsiness Detection System

A real-time computer vision project that detects driver fatigue using eye aspect ratio (EAR) and triggers an alert when drowsiness is detected.

🧠 Project Overview

Driver drowsiness is one of the major causes of road accidents.
This project uses computer vision, dlib, and OpenCV to detect whether a driver's eyes are closing for a prolonged time and raises an alarm to prevent accidents.

✨ Features

👁️ Real-time eye blink detection

🔍 Uses Eye Aspect Ratio (EAR) to identify drowsiness

🚨 Alarm system when the driver’s eyes remain closed

🔧 Uses dlib’s 68 facial landmark detector

📷 Works with webcam feed

⏱ Fast and accurate detection

🛠️ Tech Stack

->Python

->OpenCV

->dlib

->imutils

scipy (distance calculation)

📥 Installation
1. Clone the repository
git clone https://github.com/Kapilsks/Drowsiness-Detection-using-Dlib-.git
cd Driver-Drowsiness-Detection

2. Install dependencies
3. 
4. Download facial landmark model


▶️ How to Run

Run the script:

python drowsiness_detection.py


🧩 How It Works

Webcam captures the driver's face

dlib detects facial landmarks

EAR (Eye Aspect Ratio) is calculated:

If EAR < threshold (e.g., 0.25)

And eyes remain closed for some frames

System shows Alert 

Alarm alerts the driver to wake up

📂 Project Structure
Driver-Drowsiness-Detection/
├── assets
├── drowsiness_detection.py
├── shape_predictor_68_face_landmarks.dat
├── README.md
