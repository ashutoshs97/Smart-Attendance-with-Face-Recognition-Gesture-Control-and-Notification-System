from flask import Flask, render_template, request, jsonify
from datetime import datetime
import threading
import csv
import os

app = Flask(__name__)
attendance_records = []
attendance_lock = threading.Lock()
csv_file_path = "attendance_log.csv"
REQUIRED_FIELDS = ["Name", "Email", "Date", "Time", "Timestamp", "ImagePath"]

def load_attendance_from_csv():
    """Load existing attendance records from the CSV file, skipping malformed rows."""
    if os.path.exists(csv_file_path):
        with open(csv_file_path, "r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if all(row.get(field) for field in REQUIRED_FIELDS):
                    attendance_records.append(row)
    else:
        with open(csv_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()

@app.route('/')
def index():
    """Renders the dashboard homepage."""
    with attendance_lock:
        return render_template('index.html', attendance_records=attendance_records)

@app.route('/api/attendance', methods=['POST'])
def receive_attendance():
    """Receives attendance data from the main.py script."""
    data = request.json
    if data and all(key in data for key in REQUIRED_FIELDS):
        with attendance_lock:
            attendance_records.append(data)
            with open(csv_file_path, "a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=REQUIRED_FIELDS)
                writer.writerow(data)

        return jsonify({"status": "success"}), 200
    return jsonify({"status": "error", "message": "Missing or malformed data"}), 400

@app.route('/api/get_attendance')
def get_attendance():
    """API to get all current attendance records."""
    with attendance_lock:
        return jsonify(attendance_records)

if __name__ == '__main__':
    load_attendance_from_csv()
    app.run(debug=True)