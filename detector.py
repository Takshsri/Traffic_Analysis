import cv2
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Initialize Analytics Storage
# ----------------------------
analytics_data = []

# ----------------------------
# Load YOLO Model
# ----------------------------
model = YOLO("yolov8n.pt")

# ----------------------------
# Open Video
# ----------------------------
cap = cv2.VideoCapture("videos/traffic2.mp4")

# Video Properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# ----------------------------
# Output Video Writer
# ----------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "outputs/output.mp4",
    fourcc,
    fps,
    (width, height)
)

# ----------------------------
# Vehicle Classes
# ----------------------------
vehicle_classes = ["car", "bus", "truck", "motorcycle"]

# ----------------------------
# Colors (BGR Format)
# ----------------------------
colors = {
    "car": (255, 0, 0),          # Blue
    "bus": (0, 255, 255),        # Yellow
    "truck": (0, 0, 255),        # Red
    "motorcycle": (255, 255, 0)  # Cyan
}

# ----------------------------
# Tracking Variables
# ----------------------------
tracked_ids = set()

total_counts = {
    "car": 0,
    "bus": 0,
    "truck": 0,
    "motorcycle": 0
}

# Counting Line Position
line_y = height // 2

# Frame Counter
frame_number = 0

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Draw Counting Line
    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "Counting Line",
        (20, line_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Run Tracking
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    if results[0].boxes is not None:

        for box in results[0].boxes:

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name not in vehicle_classes:
                continue

            if box.id is None:
                continue

            track_id = int(box.id[0])

            # Bounding Box Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Center Point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Count Vehicle When Crossing Line
            if center_y > line_y and track_id not in tracked_ids:
                tracked_ids.add(track_id)
                total_counts[class_name] += 1

            color = colors[class_name]

            # Draw Bounding Box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Draw Center Point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (255, 255, 255),
                -1
            )

            # Draw Label
            label = f"{class_name} #{track_id}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ----------------------------
    # Analytics
    # ----------------------------
    vehicle_count = sum(total_counts.values())

    congestion_percentage = min(
        (vehicle_count / 50) * 100,
        100
    )

    # Traffic Status
    if vehicle_count < 10:
        status = "Low Traffic"
        status_color = (0, 255, 0)

    elif vehicle_count < 25:
        status = "Medium Traffic"
        status_color = (0, 255, 255)

    else:
        status = "Heavy Traffic"
        status_color = (0, 0, 255)

    # ----------------------------
    # Dashboard Text
    # ----------------------------
    cv2.putText(
        frame,
        f"Total Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Traffic Status: {status}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )

    cv2.putText(
        frame,
        f"Cars: {total_counts['car']}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        colors["car"],
        2
    )

    cv2.putText(
        frame,
        f"Buses: {total_counts['bus']}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        colors["bus"],
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {total_counts['truck']}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        colors["truck"],
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {total_counts['motorcycle']}",
        (20, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        colors["motorcycle"],
        2
    )

    cv2.putText(
        frame,
        f"Congestion: {congestion_percentage:.1f}%",
        (20, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ----------------------------
    # Save Analytics
    # ----------------------------
    analytics_data.append({
        "frame": frame_number,
        "total_vehicles": vehicle_count,
        "cars": total_counts["car"],
        "buses": total_counts["bus"],
        "trucks": total_counts["truck"],
        "motorcycles": total_counts["motorcycle"],
        "traffic_status": status,
        "congestion_percentage": congestion_percentage
    })

    # Save Frame
    out.write(frame)

    # Display Frame
    cv2.imshow("Traffic Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

# ----------------------------
# Save CSV Report
# ----------------------------
df = pd.DataFrame(analytics_data)
df.to_csv(
    "outputs/traffic_report.csv",
    index=False
)

# ----------------------------
# Generate Vehicle Distribution Graph
# ----------------------------
labels = list(total_counts.keys())
values = list(total_counts.values())

plt.figure(figsize=(8, 6))
plt.bar(labels, values)
plt.title("Vehicle Distribution")
plt.xlabel("Vehicle Type")
plt.ylabel("Count")
plt.savefig("outputs/vehicle_distribution.png")
plt.show()

print("Analysis Completed Successfully!")
print("Output Video Saved: outputs/output.mp4")
print("CSV Report Saved: outputs/traffic_report.csv")
print("Chart Saved: outputs/vehicle_distribution.png")