import cv2
import face_recognition
import numpy as np
import os
import mediapipe as mp
from datetime import datetime
import csv
import requests
import sys
import google_sheets_integration
import multiprocessing
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

# --- Part 1: Helper function for encoding ---
def encode_face_from_path(image_path_name_tuple):
    """A helper function to encode a single face for a single image."""
    image_path, name = image_path_name_tuple
    try:
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        return encoding, name
    except IndexError:
        return None, None

# --- Part 2: Email Configuration ---
# Replace with your actual email and app password
SENDER_EMAIL = "-----------------------"
APP_PASSWORD = "-----------------------"

def send_email_alert(recipient_email, student_name, timestamp, photo_path):
    """Sends an email with a photo attachment."""
    try:
        msg = MIMEMultipart()
        msg["Subject"] = f"Attendance Recorded for {student_name}"
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email

        text = MIMEText(f"Attendance for {student_name} recorded at {timestamp}. The attached photo confirms the entry.")
        msg.attach(text)

        with open(photo_path, "rb") as fp:
            img = MIMEImage(fp.read(), name=os.path.basename(photo_path))
        msg.attach(img)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        print(f"Email with photo sent to {recipient_email} for {student_name}.")
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}")

# --- Main execution block ---
if __name__ == '__main__':
    print("Encoding known faces using multiple CPU cores...")
    known_face_encodings = []
    known_face_names = []
    images_dir = "student_images"
    known_face_emails = {}

    if not os.path.exists(images_dir):
        print(f"Error: The directory '{images_dir}' was not found. Please create it and add student image folders.")
        exit()

    image_files = []
    for name in os.listdir(images_dir):
        student_dir = os.path.join(images_dir, name)
        if os.path.isdir(student_dir):
            try:
                with open(os.path.join(student_dir, 'info.txt'), 'r') as f:
                    email = f.read().strip()
                    known_face_emails.setdefault(name, email)
            except FileNotFoundError:
                pass
            
            for filename in os.listdir(student_dir):
                if filename.endswith((".jpg", ".png", ".jpeg")):
                    image_files.append((os.path.join(student_dir, filename), name))

    total_images = len(image_files)
    print(f"Found {total_images} images to process.")

    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for encoding.")

    with multiprocessing.Pool(processes=num_cores) as pool:
        for i, (encoding, name) in enumerate(pool.imap_unordered(encode_face_from_path, image_files)):
            if encoding is not None:
                known_face_encodings.append(encoding)
                known_face_names.append(name)
            
            progress = (i + 1) / total_images
            bar_length = 50
            filled_length = int(bar_length * progress)
            bar = '#' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write(f'\rProgress: |{bar}| {progress:.0%}')
            sys.stdout.flush()

    sys.stdout.write('\n')

    print("Encoding complete.")
    print(f"Found {len(known_face_encodings)} known face encodings.")
    print("-" * 50)

    # --- Part 3: Attendance Logging and Real-time Processing ---
    print("Starting webcam... Press 'q' to quit.")
    video_capture = cv2.VideoCapture(0)
    process_this_frame = True
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    last_capture_time = {}

    STATIC_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images')
    os.makedirs(STATIC_IMAGE_PATH, exist_ok=True)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_frame_for_hands = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(rgb_frame_for_hands)
        thumbs_up_detected = False
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                if thumb_tip.y < index_finger_tip.y and thumb_tip.y < middle_finger_tip.y:
                    thumbs_up_detected = True

        face_locations = []
        face_names = []
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches and matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            if name != "Unknown" and thumbs_up_detected:
                current_time = datetime.now().timestamp()
                if name not in last_capture_time or (current_time - last_capture_time[name]) > 3:
                    last_capture_time[name] = current_time
                    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    date = datetime.now().strftime("%Y-%m-%d")
                    time = datetime.now().strftime("%H:%M:%S")

                    # Log to Google Sheets
                    google_sheet_data = [
                        name,
                        known_face_emails.get(name, "N/A"),
                        date,
                        time,
                        timestamp_str
                    ]
                    
                    try:
                        google_sheets_integration.append_to_google_sheet("Attendance Dashboard", "Sheet1", google_sheet_data)
                    except Exception as e:
                        print(f"Failed to append to Google Sheet: {e}")

                    # Capture and save image
                    image_filename = f"{name}_{timestamp_str}.jpg"
                    image_path = os.path.join(STATIC_IMAGE_PATH, image_filename)
                    cv2.imwrite(image_path, frame)
                    relative_image_path = f"/static/images/{image_filename}"

                    # Update local dashboard
                    payload = {
                        "Name": name,
                        "Email": known_face_emails.get(name, "N/A"),
                        "Date": date,
                        "Time": time,
                        "Timestamp": timestamp_str,
                        "ImagePath": relative_image_path
                    }
                    try:
                        requests.post("http://127.0.0.1:5000/api/attendance", json=payload)
                        print(f"Attendance data sent to dashboard for {name} with image: {relative_image_path}")
                    except requests.exceptions.ConnectionError:
                        print("Error: Could not connect to the dashboard server. Is app.py running?")

                    # Send email notification
                    email = known_face_emails.get(name, "N/A")
                    if email != "N/A":
                        send_email_alert(email, name, timestamp_str, image_path)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    hands.close()
    video_capture.release()
    cv2.destroyAllWindows()