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
# STEP 0: Detect court lines and corners.
#
# Camera sits near court level (not overhead), so in screen-space the
# sidelines converge steeply while the baseline/service lines stay
# almost flat -- and the far baseline/service line are thin, distant,
# and easily lost to noise. So instead of detecting every line directly,
# this only detects the lines that are close to the camera and reliable
# (near baseline, near service line, both sidelines), then finds the far
# baseline/service line by projective cross-ratio extrapolation along
# each sideline -- a standard single-view metrology technique: given two
# known real-world points on a line plus its vanishing point (the image
# of its point at infinity), the image position of any other point at a
# known real-world distance is determined without needing to see it.
# ============================================================

COURT_WIDTH = 8.23     # ITF singles court, meters
COURT_LENGTH = 23.77
NET_Y = COURT_LENGTH / 2
SERVICE_LINE_DIST = 6.4
NEAR_SERVICE_Y = NET_Y - SERVICE_LINE_DIST
FAR_SERVICE_Y = NET_Y + SERVICE_LINE_DIST
CENTER_X = COURT_WIDTH / 2


def _line_params(seg):
    """Infinite-line form a*x + b*y = c for a segment, plus its direction angle."""
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    norm = np.hypot(dx, dy)
    a, b = dy / norm, -dx / norm
    if a < 0 or (a == 0 and b < 0):
        a, b = -a, -b
    c = a * x1 + b * y1
    theta = np.degrees(np.arctan2(b, a))
    return a, b, c, theta


def _fit_line(points):
    vx, vy, x0, y0 = cv2.fitLine(np.array(points, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    a, b = float(vy), float(-vx)
    return a, b, a * float(x0) + b * float(y0)


def _intersect(line1, line2):
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    return ((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det)


def _cluster_flat_lines(segments, gap=25):
    """Cluster near-horizontal segments by their mean y (they barely drift
    in y along their own length, so position-based clustering is enough)."""
    items = sorted((((s[1] + s[3]) / 2, s) for s in segments), key=lambda t: t[0])
    clusters = []
    for pos, seg in items:
        if clusters and pos - clusters[-1]["positions"][-1] <= gap:
            clusters[-1]["segs"].append(seg)
            clusters[-1]["positions"].append(pos)
        else:
            clusters.append({"segs": [seg], "positions": [pos]})
    for c in clusters:
        pts = [(x, y) for s in c["segs"] for x, y in [(s[0], s[1]), (s[2], s[3])]]
        c["mean_pos"] = sum(c["positions"]) / len(c["positions"])
        c["length"] = sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in c["segs"])
        c["min_x"] = min(p[0] for p in pts)
        c["max_x"] = max(p[0] for p in pts)
    return clusters


def _cluster_diagonal_lines(segments, theta_gap=8, rho_gap=30):
    """Cluster diagonal/steep segments by their infinite-line equation
    (theta, rho), not position -- a single sideline can be split into
    fragments at very different x/y by Hough, so position-based
    clustering would wrongly treat them as separate lines."""
    items = sorted((_line_params(s) + (s,) for s in segments), key=lambda t: (t[3], t[2]))
    clusters = []
    for a, b, c, theta, seg in items:
        length = np.hypot(seg[2] - seg[0], seg[3] - seg[1])
        placed = False
        for cl in clusters:
            if abs(theta - cl["theta_ref"]) < theta_gap and abs(c - cl["c_ref"]) < rho_gap:
                total = cl["weight"] + length
                cl["theta_ref"] = (cl["theta_ref"] * cl["weight"] + theta * length) / total
                cl["c_ref"] = (cl["c_ref"] * cl["weight"] + c * length) / total
                cl["weight"] = total
                cl["segs"].append(seg)
                placed = True
                break
        if not placed:
            clusters.append({"segs": [seg], "theta_ref": theta, "c_ref": c, "weight": length})
    for cl in clusters:
        pts = [(x, y) for s in cl["segs"] for x, y in [(s[0], s[1]), (s[2], s[3])]]
        cl["line"] = _fit_line(pts)
        cl["length"] = sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in cl["segs"])
    return clusters


def _mobius_extrapolate(near_param, service_param, vanish_param, real_target):
    """Project a point at real-world distance `real_target` along a line,
    given the pixel positions of two known real points (0 and
    NEAR_SERVICE_Y) and the line's vanishing point (real distance = infinity).
    """
    gamma = (service_param - near_param) / (NEAR_SERVICE_Y * (vanish_param - service_param))
    alpha = vanish_param * gamma
    return (alpha * real_target + near_param) / (gamma * real_target + 1)


def _find_court_blob(frame):
    """Locate the court surface by scanning hue bins -- how much of the
    frame is court vs. foreground apron/background varies with camera
    placement, so a fixed sample box isn't reliable -- and scoring each
    candidate blob by how rectangular, wide, and low-in-frame it is, which
    picks the court out from similarly-colored sky or foliage."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colorful = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 80)).astype(np.uint8) * 255
    hist = cv2.calcHist([hsv[:, :, 0]], [0], colorful, [30], [0, 180]).flatten()
    order = np.argsort(hist)[::-1]

    kernel = np.ones((7, 7), np.uint8)
    best_score, best_contour = -1, None
    for bin_idx in order[:8]:
        if hist[bin_idx] < h * w * 0.005:
            continue
        center_hue = bin_idx * 6 + 3
        lower = np.array([max(0, center_hue - 10), 60, 80])
        upper = np.array([min(179, center_hue + 10), 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        x, y, bw, bh = cv2.boundingRect(largest)
        if y < h * 0.15:  # touches the sky/background band -> not the court
            continue
        rectangularity = area / (bw * bh + 1e-6)
        width_frac = bw / w
        score = area * rectangularity * width_frac
        if score > best_score:
            best_score, best_contour = score, largest

    if best_contour is None:
        raise RuntimeError("Could not find a court-colored region on the first frame")
    return best_contour


def _lines_only_mask(frame, court_contour):
    h, w = frame.shape[:2]
    kernel = np.ones((7, 7), np.uint8)
    court_region_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(court_region_mask, [court_contour], -1, 255, cv2.FILLED)
    court_region_mask = cv2.dilate(court_region_mask, kernel, iterations=1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(white_thresh, white_thresh, mask=court_region_mask)


def detect_court_points(frame):
    """Detect tennis court reference points on a (static-camera) frame.
    Returns a dict of named pixel points, always including
    BL/BR/TL/TR (outer corners), plus NSL/NSR/FSL/FSR (service line
    corners) and NCT (near center-service T) when confidently found."""
    h, w = frame.shape[:2]
    court_contour = _find_court_blob(frame)
    lines_mask = _lines_only_mask(frame, court_contour)

    edges = cv2.Canny(lines_mask, 50, 150)
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=int(w * 0.06), maxLineGap=25)
    if segments is None:
        raise RuntimeError("No court lines detected -- check lighting/mask in the debug image")
    segments = segments.reshape(-1, 4)

    flats, diagonals = [], []
    for seg in segments:
        x1, y1, x2, y2 = seg
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        dist_from_horizontal = min(angle, 180 - angle)  # 0=flat, 90=vertical
        (flats if dist_from_horizontal < 8 else diagonals).append(seg)

    h_clusters = [c for c in _cluster_flat_lines(flats) if c["length"] > w * 0.15]
    h_clusters.sort(key=lambda c: -c["mean_pos"])  # nearest (largest pixel y) first
    if len(h_clusters) < 2:
        raise RuntimeError(f"Need a near baseline + near service line, found {len(h_clusters)} usable flat lines")
    near_baseline_c, near_service_c = h_clusters[0], h_clusters[1]
    near_baseline_l = _fit_line([(x, y) for s in near_baseline_c["segs"] for x, y in [(s[0], s[1]), (s[2], s[3])]])
    near_service_l = _fit_line([(x, y) for s in near_service_c["segs"] for x, y in [(s[0], s[1]), (s[2], s[3])]])

    v_clusters = [c for c in _cluster_diagonal_lines(diagonals) if c["length"] > h * 0.08]

    # Disambiguate real sidelines from same-colored clutter (e.g. a neighboring
    # court) by how far their intersection with the near baseline extrapolates
    # beyond the baseline's own visible extent -- the true sideline should be
    # a small extrapolation, not a big jump.
    left_candidates, right_candidates = [], []
    for c in v_clusters:
        pt = _intersect(near_baseline_l, c["line"])
        if pt is None:
            continue
        if pt[0] < near_baseline_c["min_x"]:
            left_candidates.append((near_baseline_c["min_x"] - pt[0], c, pt))
        elif pt[0] > near_baseline_c["max_x"]:
            right_candidates.append((pt[0] - near_baseline_c["max_x"], c, pt))
    if not left_candidates or not right_candidates:
        raise RuntimeError("Could not find both sidelines")
    _, left_c, BL = min(left_candidates, key=lambda t: t[0])
    _, right_c, BR = min(right_candidates, key=lambda t: t[0])
    left_l, right_l = left_c["line"], right_c["line"]

    NSL = _intersect(near_service_l, left_l)
    NSR = _intersect(near_service_l, right_l)
    vanish = _intersect(left_l, right_l)
    if vanish is None:
        raise RuntimeError("Sidelines came out parallel -- detection failed")

    def far_point(near_pt, service_pt, line, real_target):
        y = _mobius_extrapolate(near_pt[1], service_pt[1], vanish[1], real_target)
        a, b, c = line
        return (c - b * y) / a, y

    points = {
        "BL": BL, "BR": BR,
        "NSL": NSL, "NSR": NSR,
        "TL": far_point(BL, NSL, left_l, COURT_LENGTH),
        "TR": far_point(BR, NSR, right_l, COURT_LENGTH),
        "FSL": far_point(BL, NSL, left_l, FAR_SERVICE_Y),
        "FSR": far_point(BR, NSR, right_l, FAR_SERVICE_Y),
    }

    # Bonus point: the short center-service line, searched for with a finer
    # Hough pass in a small window since it's much shorter than the other
    # lines and gets lost against the threshold/minLineLength tuned for them.
    cx = (NSL[0] + NSR[0]) / 2
    top = int(min(NSL[1], NSR[1]) - (near_baseline_c["mean_pos"] - near_service_c["mean_pos"]))
    roi = lines_mask[max(0, top):int(near_service_c["mean_pos"]) + 5,
                      max(0, int(cx - 60)):int(cx + 60)]
    if roi.size:
        roi_edges = cv2.Canny(roi, 50, 150)
        roi_segments = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=15)
        if roi_segments is not None:
            x_off, y_off = max(0, int(cx - 60)), max(0, top)
            center_pts = []
            for seg in roi_segments.reshape(-1, 4):
                x1, y1, x2, y2 = seg
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                if min(angle, 180 - angle) > 60:
                    center_pts += [(x1 + x_off, y1 + y_off), (x2 + x_off, y2 + y_off)]
            if len(center_pts) >= 2:
                center_l = _fit_line(center_pts)
                points["NCT"] = _intersect(near_service_l, center_l)

    debug_img = frame.copy()
    for c, color in [(near_baseline_c, (0, 255, 255)), (near_service_c, (0, 255, 255))]:
        for seg in c["segs"]:
            cv2.line(debug_img, (seg[0], seg[1]), (seg[2], seg[3]), color, 2)
    for c in (left_c, right_c):
        for seg in c["segs"]:
            cv2.line(debug_img, (seg[0], seg[1]), (seg[2], seg[3]), (255, 0, 255), 2)
    for name, pt in points.items():
        if pt is None:
            continue
        px, py = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(debug_img, (px, py), 8, (0, 0, 255), -1)
        cv2.putText(debug_img, name, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return points, debug_img


ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame for court detection")

court_points, court_debug_img = detect_court_points(first_frame)
for corner in ("BL", "BR", "TL", "TR"):
    if court_points.get(corner) is None:
        raise RuntimeError(f"Failed to detect court corner {corner} -- see court_detection_debug.png")

cv2.imwrite("court_detection_debug.png", court_debug_img)
cv2.imshow("Court detection (press any key to continue)", court_debug_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

BL, BR, TR, TL = court_points["BL"], court_points["BR"], court_points["TR"], court_points["TL"]

# save all detected court points to their own CSV (columns for points that
# weren't confidently detected, e.g. NCT, are left blank)
court_point_names = ["BL", "BR", "TR", "TL", "NSL", "NSR", "FSL", "FSR", "NCT"]
court_row = {}
for name in court_point_names:
    pt = court_points.get(name)
    court_row[f"{name}_x"] = [pt[0] if pt else None]
    court_row[f"{name}_y"] = [pt[1] if pt else None]
court_row["fps"] = [fps]  # so Mapping.py can play back at the source video's real speed
court_df = pd.DataFrame(court_row)

k = 1
while os.path.exists(f"court_coordinates_{k}.csv"):
    k += 1
court_df.to_csv(f'court_coordinates_{k}.csv', index=False)
print(f"Saved court corners to court_coordinates_{k}.csv")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
