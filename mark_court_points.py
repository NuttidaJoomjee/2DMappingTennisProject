"""Manually mark court reference points on a video's first frame, as a
precise alternative to automatic detection in OpenVCPipline.py.

For each point: click once on the full frame to get roughly the right spot,
then click again on the zoomed-in view that pops up to place it exactly.
Press 'u' at any time to undo the last confirmed point and redo it.

Produces a court_coordinates_N.csv with one column per point above (plus
the original 9 detect_court_points() also produces, so Mapping.py's
homography fit -- which only reads BL/BR/TL/TR/NSL/NSR/FSL/FSR -- keeps
working unchanged) -- pair it with whichever existing
player_tracking_output_*.csv / detected_video_*.mp4 you're already using
(those don't depend on how the court was calibrated, only on the raw video).

The extra points below (doubles sidelines, net, center marks, far center-T)
aren't required by anything downstream yet -- they're captured for
reference/visualization now, in case a future feature (e.g. drawing the
doubles alley or net line in Mapping.py's 2D view) wants them. Doubles
sideline points are marked optional because this project's footage doesn't
paint a doubles alley at all (verified directly on-frame) -- skip them
unless a specific video actually shows one.
"""
import cv2
import os
import pandas as pd

VIDEO_PATH = r"D:\TennisProject\Game2.mp4"

# The original 9 detect_court_points() produces, plus every other line
# intersection a full tennis court actually has (doubles sidelines, net,
# baseline center marks, the far service line's center-T) -- grouped
# near-to-far, left-to-right at each line, matching how you'd naturally
# scan the frame.
POINT_SPECS = [
    ("NCOL", "Near camera outside Left"),
    ("NCOR", "Near camera outside Right"),
    ("BDL", "Near baseline x Left DOUBLES sideline (optional -- press 's' to skip if no doubles line is painted)"),
    ("BL", "Near baseline x Left singles sideline "),
    ("NCM", "Near baseline center mark"),
    ("BR", "Near baseline x Right singles sideline "),
    ("BDR", "Near baseline x Right DOUBLES sideline"),
    ("NSL", "Near service line x Left singles sideline"),
    ("NCT", "Near center-T: near service line x center line "),
    ("NSR", "Near service line x Right singles sideline"),
    ("NetL", "Net x Left post "),
    ("NetSL", "Net x Left singles sideline "),
    ("NetC", "Net x Center strap, the net's lowest point "),
    ("NetSR", "Net x Right singles sideline "),
    ("NetR", "Net x Right post"),
    ("FSL", "Far service line x Left singles sideline"),
    ("FCT", "Far center-T: far service line x center line "),
    ("FSR", "Far service line x Right singles sideline"),
    ("TDL", "Far baseline x Left DOUBLES sideline"),
    ("TL", "Far baseline x Left singles sideline"),
    ("FCM", "Far baseline center mark "),
    ("TR", "Far baseline x Right singles sideline"),
    ("TDR", "Far baseline x Right DOUBLES sideline"),
    ("FCOL", "Far camera outside Left"),
    ("FCOR", "Far camera outside Right"),
]

ZOOM_SCALE = 6
ZOOM_HALF = 40  # pixels of original-resolution context shown around the rough click


def _pick_rough_point(window_name, base_img):
    clicked = {}

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (x, y)

    cv2.setMouseCallback(window_name, on_mouse)
    while "pt" not in clicked:
        cv2.imshow(window_name, base_img)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("s"):
            return None
        if key == 27:
            raise KeyboardInterrupt
    return clicked["pt"]


def _pick_precise_point(frame, rough_xy):
    rx, ry = rough_xy
    h, w = frame.shape[:2]
    x0, y0 = max(0, rx - ZOOM_HALF), max(0, ry - ZOOM_HALF)
    x1, y1 = min(w, rx + ZOOM_HALF), min(h, ry + ZOOM_HALF)
    crop = frame[y0:y1, x0:x1]
    zoomed = cv2.resize(crop, (crop.shape[1] * ZOOM_SCALE, crop.shape[0] * ZOOM_SCALE),
                         interpolation=cv2.INTER_CUBIC)

    win = "Zoom in - click the EXACT point (u=undo rough pick, s=skip point)"
    cv2.namedWindow(win)
    clicked = {}

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (x, y)

    cv2.setMouseCallback(win, on_mouse)
    while True:
        display = zoomed.copy()
        # crosshair through the zoomed image center as a guide
        cv2.line(display, (display.shape[1] // 2, 0), (display.shape[1] // 2, display.shape[0]), (0, 255, 0), 1)
        cv2.line(display, (0, display.shape[0] // 2), (display.shape[1], display.shape[0] // 2), (0, 255, 0), 1)
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF
        if "pt" in clicked:
            break
        if key == ord("u"):
            cv2.destroyWindow(win)
            return "undo"
        if key == ord("s"):
            cv2.destroyWindow(win)
            return "skip"
        if key == 27:
            cv2.destroyWindow(win)
            raise KeyboardInterrupt
    cv2.destroyWindow(win)
    zx, zy = clicked["pt"]
    return (x0 + zx / ZOOM_SCALE, y0 + zy / ZOOM_SCALE)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read first frame from {VIDEO_PATH}")

    window_name = "Click each point on the FULL frame (u=undo last point, s=skip current, ESC=quit)"
    cv2.namedWindow(window_name)

    points = {}
    idx = 0
    history = []
    while idx < len(POINT_SPECS):
        name, desc = POINT_SPECS[idx]
        base_img = frame.copy()
        for pname, (px, py) in points.items():
            cv2.circle(base_img, (int(px), int(py)), 6, (0, 0, 255), -1)
            cv2.putText(base_img, pname, (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(base_img, (0, 0), (base_img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(base_img, f"Click: {name} -- {desc}", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        rough = _pick_rough_point(window_name, base_img)
        if rough is None:  # skipped
            idx += 1
            continue

        result = _pick_precise_point(frame, rough)
        if result == "undo":
            if history:
                last_name = history.pop()
                del points[last_name]
                idx -= 1
            continue
        if result == "skip":
            idx += 1
            continue

        points[name] = result
        history.append(name)
        idx += 1

    cv2.destroyAllWindows()

    court_row = {}
    for name, _ in POINT_SPECS:
        pt = points.get(name)
        court_row[f"{name}_x"] = [pt[0] if pt else None]
        court_row[f"{name}_y"] = [pt[1] if pt else None]
    court_row["fps"] = [fps]
    court_df = pd.DataFrame(court_row)

    k = 1
    while os.path.exists(f"court_coordinates_{k}.csv"):
        k += 1
    out_path = f"court_coordinates_{k}.csv"
    court_df.to_csv(out_path, index=False)
    print(f"Saved manually-marked points to {out_path}")
    for name, _ in POINT_SPECS:
        pt = points.get(name)
        print(f"  {name}: {pt}")


if __name__ == "__main__":
    main()
