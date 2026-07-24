from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("D:\\TennisProject\\video1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ============================================================
# STEP 0: Detect court corners automatically (once, from frame 0)
# ============================================================
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame for court detection")

gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(thresh, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=150, maxLineGap=20)

print(type(lines))
print(lines.shape if lines is not None else "None")
print(lines[0] if lines is not None else "None")

horizontal_lines = []   # baselines — near 0 degrees
diagonal_pos = []       # one sideline — positive angle cluster
diagonal_neg = []       # other sideline — negative angle cluster

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line   # fixed: no [0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if abs(angle) < 10:
            horizontal_lines.append((x1, y1, x2, y2, length))
        elif 10 <= angle < 30:
            diagonal_pos.append((x1, y1, x2, y2, length))
        elif -30 < angle <= -10:
            diagonal_neg.append((x1, y1, x2, y2, length))

horizontal_lines.sort(key=lambda l: -l[4])
diagonal_pos.sort(key=lambda l: -l[4])
diagonal_neg.sort(key=lambda l: -l[4])

print(f"Horizontal (baseline candidates): {len(horizontal_lines)}")
print(f"Diagonal positive (sideline candidate): {len(diagonal_pos)}")
print(f"Diagonal negative (sideline candidate): {len(diagonal_neg)}")

def pick_extreme_lines(lines_list, axis_index):
    if len(lines_list) < 2:
        return None, None
    sorted_by_pos = sorted(lines_list, key=lambda l: (l[axis_index] + l[axis_index+2]) / 2)
    return sorted_by_pos[0], sorted_by_pos[-1]

def line_intersection(line1, line2):
    if line1 is None or line2 is None:
        return None
    x1, y1, x2, y2 = line1[:4]
    x3, y3, x4, y4 = line2[:4]
    A1, B1 = y2 - y1, x1 - x2
    C1 = A1*x1 + B1*y1
    A2, B2 = y4 - y3, x3 - x4
    C2 = A2*x3 + B2*y3
    det = A1*B2 - A2*B1
    if det == 0:
        return None
    x = (B2*C1 - B1*C2) / det
    y = (A1*C2 - A2*C1) / det
    return (x, y)

left_line = diagonal_neg[0] if diagonal_neg else None
right_line = diagonal_pos[0] if diagonal_pos else None

top_line, bottom_line = pick_extreme_lines(horizontal_lines[:10], 1)

TL = line_intersection(top_line, left_line)
TR = line_intersection(top_line, right_line)
BL = line_intersection(bottom_line, left_line)
BR = line_intersection(bottom_line, right_line)

print("BL:", BL, "BR:", BR, "TR:", TR, "TL:", TL)

# save court corners to their own CSV
court_df = pd.DataFrame({
    'BL_x': [BL[0] if BL else None], 'BL_y': [BL[1] if BL else None],
    'BR_x': [BR[0] if BR else None], 'BR_y': [BR[1] if BR else None],
    'TR_x': [TR[0] if TR else None], 'TR_y': [TR[1] if TR else None],
    'TL_x': [TL[0] if TL else None], 'TL_y': [TL[1] if TL else None],
})
k = 1
while os.path.exists(f"court_coordinates_{k}.csv"):
    k += 1
court_df.to_csv(f'court_coordinates_{k}.csv', index=False)
print(f"Saved court corners to court_coordinates_{k}.csv")

# rewind video back to frame 0, since we already consumed frame 0 above
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ============================================================
# STEP 1: Player + ball tracking (your existing pipeline)
# ============================================================

# Training more tennis ball data to improve the model
# model.train(
#     data="C:\Users\ACER\Downloads\tennis ball.v1i.yolov8\data.yaml",
#     epochs=50,
#     imgsz=640,
#     batch=16
# )

i = 1
while os.path.exists(f"detected_video_{i}.mp4"):
    i += 1
output_filename = f"detected_video_{i}.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

records = []
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    player_results = model.track(frame, conf=0.25, tracker="custom_tracker.yaml", verbose=False, classes=[0])
    ball_results = model.track(frame, conf=0.1, verbose=False, classes=[32])
    annotated_frame = player_results[0].plot()
    annotated_frame = ball_results[0].plot(img=annotated_frame)

    # draw the detected court corners on every frame too, for visual reference
    for pt in [BL, BR, TR, TL]:
        if pt:
            cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)

    out.write(annotated_frame)

    row = {'frame': frame_num}

    if player_results[0].boxes.id is not None:
        boxes = player_results[0].boxes.xyxy.cpu().numpy()
        track_ids = player_results[0].boxes.id.cpu().numpy()
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            foot_x = (x1 + x2) / 2
            foot_y = y2
            row[f'player_{int(track_id)}_x'] = foot_x
            row[f'player_{int(track_id)}_y'] = foot_y

    if len(ball_results[0].boxes) > 0:
        box = ball_results[0].boxes.xyxy.cpu().numpy()[0]
        x1, y1, x2, y2 = box
        ball_x = (x1 + x2) / 2
        ball_y = (y1 + y2) / 2
        row['ball_x'] = ball_x
        row['ball_y'] = ball_y

    records.append(row)

    frame_num += 1
    if frame_num % 50 == 0:
        print(f"Processed {frame_num} frames...")

cap.release()
out.release()
print(f"Done! Saved as {output_filename}")

df = pd.DataFrame(records)
j = 1
while os.path.exists(f"player_tracking_output_{j}.csv"):
    j += 1
df.to_csv(f'player_tracking_output_{j}.csv', index=False)
print(f"Saved {len(df)} rows to player_tracking_output_{j}.csv")