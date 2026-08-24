"""
Hard-negative mining for the tennis-ball detector.

video1.mp4 is one continuous rally (no dead frames), so full frames can't be
safely vetted as ball-free by eye -- the ball could be anywhere. Instead this
crops tight patches directly around confirmed false-positive fixture
locations (found via the same recurring-static-cell logic used at inference
time in OpenVCPipline.py). The ball's trajectory never reaches into the tree
line where these fixtures sit, so a crop centered there is guaranteed clean,
regardless of what the rest of the frame is doing.

Saved as background images (empty label file) directly into the training
set at the path ball_dataset/train/images points to.
"""
from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("video1.mp4")

STATIC_GRID = 12
STATIC_HIT_DECAY = 0.97
STATIC_HIT_THRESHOLD = 20
static_hits = {}
static_blacklist = set()


def cell(cx, cy):
    return (int(cx) // STATIC_GRID, int(cy) // STATIC_GRID)


def register(cx, cy):
    for c in list(static_hits):
        decayed = static_hits[c] * STATIC_HIT_DECAY
        if decayed < 0.01:
            del static_hits[c]
        else:
            static_hits[c] = decayed
    k = cell(cx, cy)
    static_hits[k] = static_hits.get(k, 0.0) + 1.0
    if static_hits[k] > STATIC_HIT_THRESHOLD:
        static_blacklist.add(k)


OUT_IMG_DIR = "ball_dataset/train/images"
OUT_LBL_DIR = "ball_dataset/train/labels"
CROP = 96
SAVE_EVERY = 15  # frames between crops of the same recurring fixture, for mild diversity

frame_idx = 0
saved = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    results = model.track(frame, conf=0.1, verbose=False, classes=[32], persist=True)
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            register(cx, cy)
            if cell(cx, cy) not in static_blacklist:
                continue
            if frame_idx % SAVE_EVERY != 0:
                continue
            x0 = max(0, min(w - CROP, int(cx - CROP / 2)))
            y0 = max(0, min(h - CROP, int(cy - CROP / 2)))
            crop = frame[y0:y0 + CROP, x0:x0 + CROP]
            if crop.shape[0] != CROP or crop.shape[1] != CROP:
                continue
            name = f"hardneg_video1_f{frame_idx}_{int(cx)}_{int(cy)}"
            cv2.imwrite(os.path.join(OUT_IMG_DIR, name + ".jpg"), crop)
            open(os.path.join(OUT_LBL_DIR, name + ".txt"), "w").close()
            saved.append(name)
    frame_idx += 1

print(f"Blacklisted cells: {sorted(static_blacklist)}")
print(f"Saved {len(saved)} hard-negative crops to {OUT_IMG_DIR}")
for n in saved:
    print(" ", n)
