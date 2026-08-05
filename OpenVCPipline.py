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

crop_start_y = 10000   # adjust based on where the court actually starts in your frame
gray_cropped = gray[crop_start_y:, :]


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

# --- BL/BR: automated, as before ---
#The top line is having a problem but the bottom line is fine so we keep it work normally
BL = line_intersection(bottom_line, left_line)
BR = line_intersection(bottom_line, right_line)
print("BL:", BL, "BR:", BR)

bl_x, bl_y = BL
br_x, br_y = BR


near_baseline_width = abs(br_x - bl_x)
near_baseline_y = (bl_y + br_y) / 2

search_y_min = near_baseline_y * 0.45
search_y_max = near_baseline_y * 0.65

candidate_far_lines = [
    l for l in horizontal_lines
    if search_y_min <= (l[1] + l[3]) / 2 <= search_y_max
    and l[4] < near_baseline_width * 0.8
]

# CHANGED: pick by largest average y (closest to camera among candidates), not longest
candidate_far_lines.sort(key=lambda l: -((l[1] + l[3]) / 2))
top_line = candidate_far_lines[0] if candidate_far_lines else None

print(f"Far baseline candidates in range: {len(candidate_far_lines)}")
print("Selected top_line:", top_line)

TL = line_intersection(top_line, left_line)
TR = line_intersection(top_line, right_line)

# ============================================================
# Visual verification
# ============================================================
line_check_frame = first_frame.copy()

def draw_line(img, line, color, label):
    if line is not None:
        x1, y1, x2, y2 = line[:4]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

for l in candidate_far_lines:
    x1, y1, x2, y2, _ = l
    cv2.line(line_check_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # green = all candidates

draw_line(line_check_frame, top_line, (0, 0, 255), 'top_line (selected)')   # red = chosen one
draw_line(line_check_frame, bottom_line, (255, 0, 0), 'bottom_line')
draw_line(line_check_frame, left_line, (0, 255, 0), 'left_line')
draw_line(line_check_frame, right_line, (0, 255, 255), 'right_line')

cv2.imshow('Lines used for corners', line_check_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("BL:", BL, "BR:", BR, "TR:", TR, "TL:", TL)

if BL is not None and TL is not None:
    print("BL_y > TL_y (should be True):", BL[1] > TL[1])


# save court corners to their own CSV
court_df = pd.DataFrame({
    'BL_x': [BL[0] if BL else None], 'BL_y': [BL[1] if BL else None],
    'BR_x': [BR[0] if BR else None], 'BR_y': [BR[1] if BR else None],
    'TR_x': [TR[0] if TR else None], 'TR_y': [TR[1] if TR else None],
    'TL_x': [TL[0] if TL else None], 'TL_y': [TL[1] if TL else None],
})

#ตั้งชื่อไฟล์ของ Court ถ้ามีไฟล์นั้นอยู่แล้วให้เพิ่มเลขไป1
k = 1
while os.path.exists(f"court_coordinates_{k}.csv"):
    k += 1
court_df.to_csv(f'court_coordinates_{k}.csv', index=False)
print(f"Saved court corners to court_coordinates_{k}.csv")

# rewind video back to frame 0, since we already consumed frame 0 above
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


#ตั้งชื่อไฟล์ของ player and ball ถ้ามีไฟล์นัั้นแล้วให้เพิ่มเลขไป1
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