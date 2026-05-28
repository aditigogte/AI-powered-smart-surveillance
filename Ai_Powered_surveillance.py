# merged_final_surveillance.py
# Final merged version (core runnable all-in-one demo)

import cv2
import os
import time
import math
from ultralytics import YOLO

# ======================
# MODELS
# ======================
person_model = YOLO("yolov8n.pt")

# Optional custom weapon model
USE_WEAPON_MODEL = False   # set True if you have weapon.pt
if USE_WEAPON_MODEL:
    weapon_model = YOLO("weapon.pt")

# ======================
# SETTINGS
# ======================
cap = cv2.VideoCapture(0)
os.makedirs("evidence", exist_ok=True)

speed_threshold = 35
restricted_zone = (100, 100, 400, 400)

track_history = {}
positions = {}
pid_counter = 0
last_alert = 0

# ======================
# FUNCTIONS
# ======================
def distance(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def assign_ids(detections, prev):
    global pid_counter
    updated = {}
    used = set()

    for det in detections:
        matched = None
        best = float("inf")

        for pid, old in prev.items():
            d = distance(det, old)
            if d < 50 and d < best and pid not in used:
                matched = pid
                best = d

        if matched is None:
            matched = pid_counter
            pid_counter += 1

        updated[matched] = det
        used.add(matched)

    return updated

def alert(frame, reason):
    global last_alert
    now = time.time()

    if now - last_alert > 3:
        filename = f"evidence/{int(now)}_{reason}.jpg"
        cv2.imwrite(filename, frame)
        print("🚨 ALERT:", reason, "| Saved:", filename)
        last_alert = now

# ======================
# MAIN LOOP
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Night Vision
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 60:
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
        cv2.putText(frame, "Night Vision ON", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Detect Persons
    results = person_model(frame, verbose=False)

    detections = []
    boxes = []
    count = 0

    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])

            if cls == 0:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                cx,cy = (x1+x2)//2, (y1+y2)//2

                detections.append((cx,cy))
                boxes.append((x1,y1,x2,y2,cx,cy))
                count += 1

    positions = assign_ids(detections, positions)

    # Restricted Zone
    rx1,ry1,rx2,ry2 = restricted_zone
    cv2.rectangle(frame,(rx1,ry1),(rx2,ry2),(255,0,0),2)
    cv2.putText(frame,"Restricted Zone",(rx1,ry1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    # History
    for pid, center in positions.items():
        if pid not in track_history:
            track_history[pid] = []

        track_history[pid].append(center)

        if len(track_history[pid]) > 10:
            track_history[pid].pop(0)

    # Analyze Persons
    for (x1,y1,x2,y2,cx,cy) in boxes:
        matched_id = None

        for pid, pos in positions.items():
            if abs(pos[0]-cx) < 30 and abs(pos[1]-cy) < 30:
                matched_id = pid
                break

        if matched_id is None:
            continue

        suspicious = False
        reason = ""
        color = (0,255,0)

        hist = track_history[matched_id]

        # Running
        if len(hist) >= 2:
            speed = distance(hist[-2], hist[-1])
            if speed > speed_threshold:
                suspicious = True
                reason = "Running"

        # Intrusion
        if rx1 < cx < rx2 and ry1 < cy < ry2:
            suspicious = True
            reason = "Intrusion"

        # Loitering
        if len(hist) >= 8:
            movement = sum(distance(hist[i-1], hist[i]) for i in range(1,len(hist)))
            if movement < 20:
                suspicious = True
                reason = "Loitering"

        label = f"ID {matched_id}"

        if suspicious:
            color = (0,0,255)
            label += " ALERT: " + reason
            alert(frame, reason)

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # Fighting Detection
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            c1 = (boxes[i][4], boxes[i][5])
            c2 = (boxes[j][4], boxes[j][5])

            if distance(c1,c2) < 100:
                cv2.line(frame,c1,c2,(0,0,255),2)
                cv2.putText(frame,"Possible Fight",
                            (min(c1[0],c2[0]), min(c1[1],c2[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
                alert(frame, "Fight")

    # Weapon Detection
    if USE_WEAPON_MODEL:
        w_results = weapon_model(frame, verbose=False)

        for r in w_results:
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])

                if conf > 0.5:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                    cv2.putText(frame,"WEAPON DETECTED",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    alert(frame, "Weapon")

    # Crowd Count
    cv2.putText(frame, f"People Count: {count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

    cv2.imshow("Merged Final AI Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()