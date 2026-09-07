from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd

model = YOLO("yolov8n.pt")

pose_model = YOLO("yolov8n-pose.pt")
LEFT_ANKLE, RIGHT_ANKLE = 15, 16  # COCO keypoint indices


ball_model = model

cap = cv2.VideoCapture("D:\\TennisProject\\Game2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# STEP 0: Detect court lines and corners.

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


def _mobius_extrapolate(near_param, ref_param, ref_real_y, vanish_param, real_target):
    gamma = (ref_param - near_param) / (ref_real_y * (vanish_param - ref_param))
    alpha = vanish_param * gamma
    return (alpha * real_target + near_param) / (gamma * real_target + 1)


def _find_court_blob(frame):
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
    # The raw color-blob contour can have a ragged hole in it (e.g. a
    # shadow/lighting band across the service-line area shifts hue enough to
    # drop out of range), which turns into a real gap in the mask below and
    # silently excludes real lines that fall inside that gap. The true court
    # region is always a convex quadrilateral in image space regardless of
    # camera angle, so taking the convex hull closes gaps like that robustly.
    return cv2.convexHull(best_contour)


def _lines_only_mask(frame, court_contour):
    h, w = frame.shape[:2]
    kernel = np.ones((7, 7), np.uint8)
    court_region_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(court_region_mask, [court_contour], -1, 255, cv2.FILLED)
    court_region_mask = cv2.dilate(court_region_mask, kernel, iterations=1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A fixed brightness cutoff only works for the exposure/lighting of one
    # specific video -- a different camera can render "white paint" at a much
    # lower absolute gray value. Calibrate the cutoff instead from the
    # court region's own brightness distribution (lines are the bright tail
    # against the comparatively uniform, darker court surface), so it adapts
    # to whatever footage is loaded.
    court_pixels = gray[court_region_mask > 0].astype(np.float64)
    if court_pixels.size:
        court_mean, court_std = court_pixels.mean(), court_pixels.std()
    else:
        court_mean, court_std = 180.0, 0.0
    thresh_val = np.clip(court_mean + 2.0 * court_std, 120, 220)
    _, white_thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(white_thresh, white_thresh, mask=court_region_mask), court_mean, court_std


def _find_faint_flat_line(frame, court_mean, court_std, y0, y1, x0, x1, min_length_frac=0.15):
    
    h, w = frame.shape[:2]
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    if y1 - y0 < 5 or x1 - x0 < 5:
        return None
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    thresh_val = np.clip(court_mean + 1.0 * court_std, 100, 200)
    _, band_thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(band_thresh, 50, 150)
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                minLineLength=int((x1 - x0) * 0.1), maxLineGap=40)
    if segments is None:
        return None
    flats = []
    for seg in segments.reshape(-1, 4):
        sx1, sy1, sx2, sy2 = seg
        angle = np.degrees(np.arctan2(sy2 - sy1, sx2 - sx1)) % 180
        if min(angle, 180 - angle) < 8:
            flats.append((sx1 + x0, sy1 + y0, sx2 + x0, sy2 + y0))
    clusters = [c for c in _cluster_flat_lines(flats) if c["length"] > (x1 - x0) * min_length_frac]
    if not clusters:
        return None
    return max(clusters, key=lambda c: c["length"])


def _scan_sideline_edge_points(frame, line, y_values, side, search_half_width=70):
    """Directly measure the sideline's true x at each given y, instead of
    trusting a Hough-fit line's extrapolation there. The court surface is
    darker than the lighter out-of-bounds apron just beyond the sideline,
    so scanning a single row for that brightness step gives the edge's
    exact position -- verified against this project's own data to be far
    more reliable than the Hough-fit line for the far court specifically
    (a fitted line that was collinear-by-construction with a well-verified
    near point was still ~30-90px off from this measurement, since nothing
    near the far court had constrained its angle at all; this measures the
    far court directly instead of extrapolating a guess about it).
    `line` is used only to know roughly where to search at each row, not
    trusted for the actual position. `side` is "left" or "right": for the
    right sideline the court is to the left of the true edge (so the edge
    is the leftmost pixel where brightness steps up to the apron); for the
    left sideline it's the mirror image."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    a, b, c = line
    points = []
    for y in y_values:
        y = int(round(y))
        if not (0 <= y < h) or abs(a) < 1e-9:
            continue
        # line is a*x + b*y = c (fit_line's convention) -> x = (c - b*y)/a
        predicted_x = (c - b * y) / a
        x0 = max(0, int(predicted_x - search_half_width))
        x1 = min(w, int(predicted_x + search_half_width))
        if x1 - x0 < 5:
            continue
        row = gray[y, x0:x1].astype(np.float64)
        thresh = row.mean() + 1.5 * row.std()
        bright_idx = np.where(row > thresh)[0]
        if len(bright_idx) == 0:
            continue
        edge = bright_idx.min() if side == "right" else bright_idx.max()
        points.append((x0 + edge, y))
    return points


def detect_court_points(frame):
    """Detect tennis court reference points on a (static-camera) frame.
    Returns a dict of named pixel points, always including
    BL/BR/TL/TR (outer corners), plus NSL/NSR/FSL/FSR (service line
    corners) and NCT (near center-service T) when confidently found."""
    h, w = frame.shape[:2]
    court_contour = _find_court_blob(frame)
    lines_mask, court_mean, court_std = _lines_only_mask(frame, court_contour)

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

    baseline_center_x = (near_baseline_c["min_x"] + near_baseline_c["max_x"]) / 2
    left_candidates, right_candidates, all_candidates = [], [], []
    for c in v_clusters:
        pt = _intersect(near_baseline_l, c["line"])
        if pt is None:
            continue
        all_candidates.append((pt, c))
        if pt[0] < near_baseline_c["min_x"]:
            left_candidates.append((near_baseline_c["min_x"] - pt[0], c, pt))
        elif pt[0] > near_baseline_c["max_x"]:
            right_candidates.append((pt[0] - near_baseline_c["max_x"], c, pt))
    if not left_candidates or not right_candidates:
        raise RuntimeError("Could not find both sidelines")

    # Default: whichever candidate's baseline-intersection sits closest to
    # the near baseline's own detected edge -- correct whenever that edge
    # is a reasonable proxy for the true corner.
    _, left_c, BL = min(left_candidates, key=lambda t: t[0])
    _, right_c, BR = min(right_candidates, key=lambda t: t[0])

    
    DOUBLES_WIDTH = 10.97  # ITF doubles court width, meters
    SINGLES_TO_DOUBLES_RATIO = COURT_WIDTH / DOUBLES_WIDTH
    RATIO_TOLERANCE = 0.10
    MIN_RELATIVE_LENGTH = 0.6

    def _refine_with_doubles_check(default_pt, default_c):
        default_offset = abs(default_pt[0] - baseline_center_x)
        default_side = default_pt[0] < baseline_center_x
        best_pt, best_c, best_offset = default_pt, default_c, default_offset
        for pt, c in all_candidates:
            if (pt[0] < baseline_center_x) != default_side:
                continue
            offset = abs(pt[0] - baseline_center_x)
            if offset >= best_offset:
                continue
            ratio = offset / default_offset
            if abs(ratio - SINGLES_TO_DOUBLES_RATIO) > RATIO_TOLERANCE:
                continue
            if c["length"] < MIN_RELATIVE_LENGTH * default_c["length"]:
                continue
            best_pt, best_c, best_offset = pt, c, offset
        return best_pt, best_c

    BL, left_c = _refine_with_doubles_check(BL, left_c)
    BR, right_c = _refine_with_doubles_check(BR, right_c)
    left_l, right_l = left_c["line"], right_c["line"]

    NSL = _intersect(near_service_l, left_l)
    NSR = _intersect(near_service_l, right_l)
    vanish = _intersect(left_l, right_l)
    if vanish is None:
        raise RuntimeError("Sidelines came out parallel -- detection failed")

    def far_point(near_pt, ref_pt, ref_real_y, line, real_target):
        y = _mobius_extrapolate(near_pt[1], ref_pt[1], ref_real_y, vanish[1], real_target)
        a, b, c = line
        return (c - b * y) / a, y

    # Initial estimate, extrapolated all the way from the two near-court
    # lines out to the far service line.
    FSL = far_point(BL, NSL, NEAR_SERVICE_Y, left_l, FAR_SERVICE_Y)
    FSR = far_point(BR, NSR, NEAR_SERVICE_Y, right_l, FAR_SERVICE_Y)

    match_tolerance = abs(near_baseline_c["mean_pos"] - vanish[1]) * 0.05
    est_y = (FSL[1] + FSR[1]) / 2
    far_service_c = next((c for c in h_clusters[2:] if abs(c["mean_pos"] - est_y) < match_tolerance), None)

    if far_service_c is not None:
        far_service_l = _fit_line([(x, y) for s in far_service_c["segs"] for x, y in [(s[0], s[1]), (s[2], s[3])]])
        FSL = _intersect(far_service_l, left_l)
        FSR = _intersect(far_service_l, right_l)
        TL = far_point(BL, FSL, FAR_SERVICE_Y, left_l, COURT_LENGTH)
        TR = far_point(BR, FSR, FAR_SERVICE_Y, right_l, COURT_LENGTH)

        # left_l/right_l's angle is normally set almost entirely by
        # near/mid-court Hough segments -- verified against this project's
        # own data, the far corners had zero pre-existing segments within
        # 200px of them, so their position out there is pure extrapolation
        # of a line nothing actually constrains that far out. A small
        # angular error is invisible near the segments that set it but
        # compounds into a large positional error by the time it reaches
        # the far corners. Directly measuring the court/apron brightness
        # edge at a series of rows, instead, matched a manual pixel-level
        # check to a few pixels -- but only when used *on its own*: merging
        # these direct measurements into the old Hough segments and
        # refitting through both together (tried first) still left the far
        # corner visibly off, diluted back down by whatever bias the old
        # segments carried. Scanning the full sideline (near court to far
        # baseline) this way and replacing the Hough fit outright, instead
        # of blending it with an independently less-reliable one, is what
        # actually closed the gap on this project's own data.
        # Restricted to the far-to-mid court, not all the way down to the
        # near baseline: scanning that far was found, on this project's
        # own data, to pick up a discontinuous jump partway down (the
        # trend from two sub-ranges implied genuinely different lines,
        # ~150px apart when extrapolated to meet) -- almost certainly the
        # adjacent court visible in frame, the same contamination already
        # seen elsewhere in this file. The near court doesn't need this
        # anyway; the existing Hough fit is already reliable there.
        scan_y_values = range(int(min(TL[1], TR[1])) - 20, int(near_service_c["mean_pos"]) - 30, 10)
        left_scan_pts = _scan_sideline_edge_points(frame, left_l, scan_y_values, "left")
        right_scan_pts = _scan_sideline_edge_points(frame, right_l, scan_y_values, "right")

        def _reject_scan_outliers(points, max_residual_px=15.0):
            # A player standing near a scanned row (their bright shoes/legs
            # against the darker court) can occasionally win the
            # brightness-step search instead of the true court edge --
            # verified against this project's own data: 2 of 22 scanned
            # points landed ~140px off a trend the other 20 agreed on
            # tightly. Checking residuals against a plain least-squares fit
            # of all the points (including those 2) doesn't reliably catch
            # this -- also verified directly: a couple of extreme-leverage
            # outliers pull an ordinary least-squares line enough that
            # *good* points start looking anomalous relative to it too, so
            # this uses DIST_HUBER (down-weights points far from the
            # emerging fit as it iterates, rather than treating every point
            # equally) just for this outlier-detection pass specifically.
            if len(points) < 5:
                return points
            pts_arr = np.array(points, dtype=np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts_arr, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            a, b = float(vy), float(-vx)
            c = a * float(x0) + b * float(y0)
            return [p for p in points if abs(a * p[0] + b * p[1] - c) < max_residual_px]

        left_scan_pts = _reject_scan_outliers(left_scan_pts)
        right_scan_pts = _reject_scan_outliers(right_scan_pts)
        # Enough scan points spanning a wide enough range stand on their
        # own -- prefer them outright over diluting them back into the
        # older, less-reliable Hough fit. Below that, there's too little
        # direct measurement to trust alone, so fold what there is into
        # the existing segments instead of discarding it.
        MIN_SCAN_POINTS, MIN_SCAN_SPAN = 8, 150
        def _span(pts):
            ys = [p[1] for p in pts]
            return max(ys) - min(ys) if ys else 0

        if len(left_scan_pts) >= MIN_SCAN_POINTS and _span(left_scan_pts) >= MIN_SCAN_SPAN:
            left_l = _fit_line(left_scan_pts)
        elif len(left_scan_pts) >= 3:
            left_l = _fit_line([(x, y) for s in left_c["segs"] for x, y in [(s[0], s[1]), (s[2], s[3])]]
                                + left_scan_pts)
        if len(right_scan_pts) >= MIN_SCAN_POINTS and _span(right_scan_pts) >= MIN_SCAN_SPAN:
            right_l = _fit_line(right_scan_pts)
        elif len(right_scan_pts) >= 3:
            right_l = _fit_line([(x, y) for s in right_c["segs"] for x, y in [(s[0], s[1]), (s[2], s[3])]]
                                 + right_scan_pts)
        if len(left_scan_pts) >= 3 or len(right_scan_pts) >= 3:
            NSL = _intersect(near_service_l, left_l)
            NSR = _intersect(near_service_l, right_l)
            FSL = _intersect(far_service_l, left_l)
            FSR = _intersect(far_service_l, right_l)
            TL = far_point(BL, FSL, FAR_SERVICE_Y, left_l, COURT_LENGTH)
            TR = far_point(BR, FSR, FAR_SERVICE_Y, right_l, COURT_LENGTH)

        baseline_est_y = (TL[1] + TR[1]) / 2
        far_service_y = min(FSL[1], FSR[1])  # the fitted/intersected line, not the raw cluster's own mean
        gap = abs(far_service_y - baseline_est_y)
        # Bound the search by where the sidelines themselves are expected at
        # this height, not by near_baseline_c/far_service_c's own detected
        # extents -- both were found to include contamination reaching far
        # outside the true court (verified against this project's own data:
        # a nearby court visible in frame produces its own lines at a
        # similar height, which a position-only y-clustering pass can't
        # tell apart from this court's real lines). Inheriting that
        # contamination into this search let unrelated segments corrupt the
        # far baseline's fitted angle -- collinear with BR/FSR by
        # construction, but visibly off from the true corner once
        # extrapolated that much further, since a small angular error
        # compounds over distance. A margin around the sidelines' own
        # projected position keeps the search to the plausible court
        # corridor regardless of what any specific cluster happened to
        # detect nearby.
        est_width = abs(TR[0] - TL[0])
        margin = max(80.0, est_width * 0.25)
        expected_left_x = (left_l[2] - left_l[1] * baseline_est_y) / left_l[0]
        expected_right_x = (right_l[2] - right_l[1] * baseline_est_y) / right_l[0]
        far_baseline_c = _find_faint_flat_line(
            frame, court_mean, court_std,
            y0=baseline_est_y - gap * 1.5 - 20,
            y1=far_service_y - gap * 0.15,
            x0=min(expected_left_x, expected_right_x) - margin,
            x1=max(expected_left_x, expected_right_x) + margin,
        )
        if far_baseline_c is not None:
            far_baseline_l = _fit_line([(x, y) for s in far_baseline_c["segs"]
                                         for x, y in [(s[0], s[1]), (s[2], s[3])]])
            TL_direct = _intersect(far_baseline_l, left_l)
            TR_direct = _intersect(far_baseline_l, right_l)
            # Sanity check against the extrapolated estimate -- guards
            # against this more sensitive search picking up an unrelated
            # faint line (a shadow, a fence rail) on a differently-shaped
            # court/venue, rather than trusting any match unconditionally.
            tolerance = max(60.0, gap * 1.5)
            if (TL_direct is not None and TR_direct is not None
                    and abs(TL_direct[1] - TL[1]) < tolerance
                    and abs(TR_direct[1] - TR[1]) < tolerance):
                TL, TR = TL_direct, TR_direct
    else:
        TL = far_point(BL, NSL, NEAR_SERVICE_Y, left_l, COURT_LENGTH)
        TR = far_point(BR, NSR, NEAR_SERVICE_Y, right_l, COURT_LENGTH)

    points = {
        "BL": BL, "BR": BR,
        "NSL": NSL, "NSR": NSR,
        "TL": TL, "TR": TR,
        "FSL": FSL, "FSR": FSR,
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
   
    for near_pt, far_pt in [(points.get("BL"), points.get("TL")), (points.get("BR"), points.get("TR"))]:
        if near_pt is not None and far_pt is not None:
            cv2.line(debug_img, (int(near_pt[0]), int(near_pt[1])), (int(far_pt[0]), int(far_pt[1])), (255, 128, 0), 1)
    for name, pt in points.items():
        if pt is None:
            continue
        px, py = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(debug_img, (px, py), 8, (0, 0, 255), -1)
        cv2.putText(debug_img, name, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return points, debug_img


def far_court_crop_box(court_points, frame_shape):
    h, w = frame_shape[:2]
    xs = [court_points[n][0] for n in ("NSL", "NSR", "TL", "TR") if court_points.get(n)]
    y_baseline = min(court_points[n][1] for n in ("TL", "TR") if court_points.get(n))
    y_near_service = max(court_points[n][1] for n in ("NSL", "NSR") if court_points.get(n))
    x0 = max(0, int(min(xs)))
    x1 = min(w, int(max(xs)))
    y0 = max(0, int(y_baseline - 130))
    y1 = min(h, int(y_near_service + 20))        # a little past the net, toward the camera
    return x0, y0, x1, y1


def detect_far_player(frame, crop_box, scale=3, ankle_conf_thresh=0.3):
   
    MAX_FAR_PLAYER_HEIGHT = 100  # original-scale px; see docstring

    x0, y0, x1, y1 = crop_box
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop_up = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

    def to_orig(local_xy):
        return (x0 + local_xy[0] / scale, y0 + local_xy[1] / scale)

    def best_far_sized_box(boxes_xyxy, confs):
        candidates = [i for i in range(len(boxes_xyxy))
                      if (boxes_xyxy[i][3] - boxes_xyxy[i][1]) / scale < MAX_FAR_PLAYER_HEIGHT]
        if not candidates:
            return None
        return max(candidates, key=lambda i: confs[i])

    # Tried dropping this plain-model pass in favor of pose_model's own
    # boxes (it detects people as part of estimating keypoints, so it looks
    # redundant) -- verified against this project's own data that this is
    # NOT a safe simplification: at this crop's actual resolution, the pose
    # model's detection head missed the far player in 40/40 test frames
    # while the plain detector still found them at ~0.84 confidence every
    # time. The two models' detection heads aren't equivalent at this scale
    # despite both being nominally "person detectors" trained on COCO, so
    # both calls stay.
    plain_result = model.predict(crop_up, conf=0.25, verbose=False, classes=[0])[0]
    plain_boxes = plain_result.boxes.xyxy.cpu().numpy()
    plain_confs = plain_result.boxes.conf.cpu().numpy()
    best_plain = best_far_sized_box(plain_boxes, plain_confs)
    if best_plain is None:
        return None
    bx1, by1, bx2, by2 = plain_boxes[best_plain]
    foot = ((bx1 + bx2) / 2, by2)  # box-bottom, same estimate used for the near player

    pose_result = pose_model.predict(crop_up, conf=0.25, verbose=False, classes=[0])[0]
    if pose_result.keypoints is not None and len(pose_result.boxes) > 0:
        pose_boxes = pose_result.boxes.xyxy.cpu().numpy()
        pose_confs = pose_result.boxes.conf.cpu().numpy()
        best_pose = best_far_sized_box(pose_boxes, pose_confs)
        if best_pose is not None:
            kxy = pose_result.keypoints.xy.cpu().numpy()[best_pose]
            kconf = pose_result.keypoints.conf.cpu().numpy()[best_pose] if pose_result.keypoints.conf is not None else None
            if kconf is not None and kconf[LEFT_ANKLE] > ankle_conf_thresh and kconf[RIGHT_ANKLE] > ankle_conf_thresh:
                foot = (kxy[LEFT_ANKLE] + kxy[RIGHT_ANKLE]) / 2
            elif kconf is not None and kconf[LEFT_ANKLE] > ankle_conf_thresh:
                foot = kxy[LEFT_ANKLE]
            elif kconf is not None and kconf[RIGHT_ANKLE] > ankle_conf_thresh:
                foot = kxy[RIGHT_ANKLE]

    return to_orig(foot)


# Court points now come from a manually-marked CSV (mark_court_points.py)
# instead of detect_court_points() -- this pipeline's job for this run is
# just to detect/track the player and ball; the court calibration is fixed
# input, not something to (re)detect here. detect_court_points() and its
# helpers stay defined above, unused, in case a future video without a
# manual CSV wants automatic detection again.
COURT_COORDINATES_PATH = r"D:\TennisProject\court_coordinates_15.csv"
_court_row = pd.read_csv(COURT_COORDINATES_PATH).iloc[0]
court_points = {}
for col in _court_row.index:
    if not col.endswith("_x"):
        continue
    name = col[:-2]
    y_col = f"{name}_y"
    if y_col not in _court_row.index:
        continue
    px, py = _court_row[col], _court_row[y_col]
    if pd.isna(px) or pd.isna(py):
        continue
    court_points[name] = (float(px), float(py))
print(f"Loaded {len(court_points)} court points from {COURT_COORDINATES_PATH}")

for corner in ("BL", "BR", "TL", "TR"):
    if corner not in court_points:
        raise RuntimeError(f"{COURT_COORDINATES_PATH} is missing required corner {corner}")

BL, BR, TR, TL = court_points["BL"], court_points["BR"], court_points["TR"], court_points["TL"]
FAR_COURT_CROP = far_court_crop_box(court_points, (height, width))


_far_y = min(TL[1], TR[1])
_near_y = max(BL[1], BR[1])
_court_h = _near_y - _far_y
_left_x = min(BL[0], TL[0])
_right_x = max(BR[0], TR[0])
_court_w = _right_x - _left_x
BALL_Y_MIN = _far_y - _court_h * 0.6    # headroom above far baseline for lobs/serves
BALL_Y_MAX = _near_y + _court_h * 0.2   # a bit below the near baseline
BALL_X_MIN = _left_x - _court_w * 0.1
BALL_X_MAX = _right_x + _court_w * 0.1


def _ball_in_plausible_region(cx, cy):
    return BALL_X_MIN <= cx <= BALL_X_MAX and BALL_Y_MIN <= cy <= BALL_Y_MAX

STATIC_GRID = 12           # px per cell -- coarser than one ball diameter
STATIC_HIT_DECAY = 0.97
STATIC_HIT_THRESHOLD = 20  # ~1s of being detected every frame, at 24-30fps
_static_hits = {}
_static_blacklist = set()


def _static_cell(cx, cy):
    return (int(cx) // STATIC_GRID, int(cy) // STATIC_GRID)


def _register_ball_candidate(cx, cy):
    for cell in list(_static_hits):
        decayed = _static_hits[cell] * STATIC_HIT_DECAY
        if decayed < 0.01:
            del _static_hits[cell]
        else:
            _static_hits[cell] = decayed
    cell = _static_cell(cx, cy)
    _static_hits[cell] = _static_hits.get(cell, 0.0) + 1.0
    if _static_hits[cell] > STATIC_HIT_THRESHOLD:
        _static_blacklist.add(cell)


def _is_static_fixture(cx, cy):
    return _static_cell(cx, cy) in _static_blacklist


# Scanning every tile across the whole plausible court region on every frame
# is the single biggest cost in this pipeline (verified: it's what turns one
# frame into 6-8 separate YOLO passes). The ball moves continuously between
# consecutive frames, so once we know roughly where it is, a small window
# around that position is enough -- only fall back to the full scan when the
# ball hasn't been seen for a few frames (lost after being occluded, leaving
# the frame, etc.) and needs to be reacquired from scratch.
_last_ball_pos = None
_frames_since_ball_seen = 999  # start "lost" so the first frame does a full scan
LOCAL_SEARCH_HALF = 260   # px around the last known position while tracking
LOST_AFTER_FRAMES = 5     # consecutive misses before reverting to a full scan


def detect_ball_tiled(frame, tile_size=640, overlap=100, conf=0.1):

    h, w = frame.shape[:2]
    x0 = max(0, int(BALL_X_MIN))
    y0 = max(0, int(BALL_Y_MIN))
    x1 = min(w, int(BALL_X_MAX))
    y1 = min(h, int(BALL_Y_MAX))

    if _last_ball_pos is not None and _frames_since_ball_seen < LOST_AFTER_FRAMES:
        lx, ly = _last_ball_pos
        tile_boxes = [(
            max(x0, int(lx - LOCAL_SEARCH_HALF)), max(y0, int(ly - LOCAL_SEARCH_HALF)),
            min(x1, int(lx + LOCAL_SEARCH_HALF)), min(y1, int(ly + LOCAL_SEARCH_HALF)),
        )]
    else:
        step = tile_size - overlap
        xs = list(range(x0, x1, step)) or [x0]
        ys = list(range(y0, y1, step)) or [y0]
        tile_boxes = [
            (max(tx, 0), max(ty, 0), min(tx + tile_size, x1, w), min(ty + tile_size, y1, h))
            for ty in ys for tx in xs
        ]

    tiles, offsets = [], []
    for tx0, ty0, tx1, ty1 in tile_boxes:
        tile = frame[ty0:ty1, tx0:tx1]
        if tile.shape[0] < 32 or tile.shape[1] < 32:
            continue
        tiles.append(tile)
        offsets.append((tx0, ty0))
    if not tiles:
        return np.empty((0, 4)), np.empty((0,))

    # One batched predict() call over all tiles instead of one call per tile
    # -- cuts the per-call pre/post-processing overhead that dominates on CPU.
    results = ball_model.predict(tiles, conf=conf, verbose=False, classes=[32])

    all_boxes, all_confs = [], []
    for r, (tx0, ty0) in zip(results, offsets):
        if len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        boxes[:, [0, 2]] += tx0
        boxes[:, [1, 3]] += ty0
        all_boxes.append(boxes)
        all_confs.append(confs)

    if not all_boxes:
        return np.empty((0, 4)), np.empty((0,))
    return np.concatenate(all_boxes), np.concatenate(all_confs)

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
    ball_boxes, ball_confs = detect_ball_tiled(frame)
    annotated_frame = player_results[0].plot()
    for bx1, by1, bx2, by2 in ball_boxes:
        cv2.rectangle(annotated_frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (255, 255, 0), 1)

    # Every point loaded from COURT_COORDINATES_PATH (up to all 21 from a
    # full manual marking pass), not just the 4 corners -- the whole point
    # of drawing them is to visually check the manual clicks against the
    # actual footage, which only works if every marked line is shown.
    for name, pt in court_points.items():
        px, py = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(annotated_frame, (px, py), 5, (0, 0, 255), -1)
        cv2.putText(annotated_frame, name, (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    row = {'frame': frame_num}

    if player_results[0].boxes.id is not None:
        boxes = player_results[0].boxes.xyxy.cpu().numpy()
        track_ids = player_results[0].boxes.id.cpu().numpy()
        #หาตำแหน่งตรงกลางของเท้า
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            foot_x = (x1 + x2) / 2
            foot_y = y2
            row[f'player_{int(track_id)}_x'] = foot_x
            row[f'player_{int(track_id)}_y'] = foot_y
            # mark the exact pixel this player's CSV row is sampled from,
            # so a mistracked box (e.g. briefly latching onto a spectator
            # instead of the player) is visible directly in the video
            cv2.circle(annotated_frame, (int(foot_x), int(foot_y)), 7, (0, 255, 255), -1)
            cv2.putText(annotated_frame, f'P{int(track_id)}', (int(foot_x) + 10, int(foot_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if len(ball_boxes) > 0:
        boxes = ball_boxes
        confs = ball_confs
        chosen = None
        for idx in np.argsort(-confs):  # highest confidence first
            x1, y1, x2, y2 = boxes[idx]
            ball_x = (x1 + x2) / 2
            ball_y = (y1 + y2) / 2
            if not _ball_in_plausible_region(ball_x, ball_y):
                continue
            # feed every plausible candidate, even ones we're about to
            # reject below, so a recurring fixture gets blacklisted fast
            _register_ball_candidate(ball_x, ball_y)
            if chosen is None and not _is_static_fixture(ball_x, ball_y):
                chosen = (ball_x, ball_y)
        if chosen is not None:
            row['ball_x'], row['ball_y'] = chosen
            cv2.circle(annotated_frame, (int(chosen[0]), int(chosen[1])), 6, (255, 0, 255), -1)
            _last_ball_pos = chosen
            _frames_since_ball_seen = 0
        else:
            _frames_since_ball_seen += 1
    else:
        _frames_since_ball_seen += 1

    far_player_pt = detect_far_player(frame, FAR_COURT_CROP)
    if far_player_pt is not None:
        row['far_player_x'], row['far_player_y'] = far_player_pt
        cv2.circle(annotated_frame, (int(far_player_pt[0]), int(far_player_pt[1])), 5, (255, 255, 0), -1)
        cv2.putText(annotated_frame, 'FAR', (int(far_player_pt[0]) + 8, int(far_player_pt[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    out.write(annotated_frame)

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
