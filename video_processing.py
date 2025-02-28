
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Global variables
lanes = []  # Stores lane points
drawing = False  # To track drawing state

def draw_lane(event, x, y, flags, param):
    """ Callback function to draw lanes using mouse clicks """
    global lanes, drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        lanes.append((x, y))
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:  # Stop drawing
        drawing = False

    elif event == cv2.EVENT_RBUTTONDOWN:  # Right-click to finish drawing
        print("Lane saved:", lanes)
        lanes = []  # Clear points after saving

# Load video
video_path = r"C:\Users\mohit\video processing\3603391451-preview.mp4"
cap = cv2.VideoCapture(video_path)

cv2.namedWindow("People Detection")
cv2.setMouseCallback("People Detection", draw_lane)

frame_rate = 10  # Adjust speed
delay = int(1000 / frame_rate)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people
    results = model(frame)
    people_count = 0
    lane_counts = {i: 0 for i in range(len(lanes))}  # Stores people count per lane

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index

            if cls == 0 and conf > 0.5:  # Class 0 = Person
                people_count += 1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Check which lane the person is in
                for i, lane in enumerate(lanes):
                    if cv2.pointPolygonTest(np.array(lane, np.int32), (center_x, center_y), False) >= 0:
                        lane_counts[i] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {people_count}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw lane polygons
    for lane in lanes:
        cv2.polylines(frame, [np.array(lane, np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    # Display total count
    cv2.putText(frame, f"Total People: {people_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display lane counts
    y_offset = 80
    for i, count in lane_counts.items():
        cv2.putText(frame, f"Lane {i + 1}: {count} people", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_offset += 30

    cv2.imshow("People Detection", frame)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        print("⏸ Paused. Press any key to continue...")
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

