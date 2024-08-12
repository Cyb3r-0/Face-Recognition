from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime, timedelta
import time

app = Flask(__name__)

# Function to encode images in the given directory
def encode_images(directory):
    known_faces = []
    known_names = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])

    return known_faces, known_names

# Function to mark attendance
def mark_attendance(name, attendance_file):
    with open(attendance_file, "a", newline='') as file:
        writer = csv.writer(file)
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([name, date_string])

# Define cap as a global variable
cap = None

# Main function for face recognition
def dream():
    global cap  # Declare cap as a global variable
    # Directory containing images of known faces
    known_faces_dir = "known_faces"
    known_faces_encodings, known_names = encode_images(known_faces_dir)

    # CSV file to store attendance
    attendance_file = "attendance.csv"
    if not os.path.isfile(attendance_file):
        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date_Time"])

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Timer variables
    start_time = time.time()
    interval = 30  # in seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Find faces in the frame
        locations = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, locations)

        for face_encoding, face_location in zip(encodings, locations):
            # Compare face with known faces
            matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                matched_index = matches.index(True)
                name = known_names[matched_index]

            # Draw rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Check if 30 seconds have passed
        if time.time() - start_time >= interval:
            for name in known_names:
                mark_attendance(name, attendance_file)
            start_time = time.time()

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Route to start the video stream
@app.route('/video_feed')
def video_feed():
    return Response(dream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to stop the video stream
@app.route('/stop_stream')
def stop_stream():
    # Add any cleanup operations here
    # For example, you can release the camera
    cap.release()
    cv2.destroyAllWindows()
    return 'Stream stopped'

# Route to serve CSV data
@app.route('/csv_data')
def csv_data():
    with open("attendance.csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
    return render_template('csv_data.html', data=data)

# Route to serve the index.html page
@app.route('/')
def index():
    return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True)
