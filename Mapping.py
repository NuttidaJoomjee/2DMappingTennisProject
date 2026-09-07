import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_rgb
import seaborn as sns
from scipy.signal import find_peaks
import cv2

court = pd.read_csv(r'D:\TennisProject\court_coordinates_15.csv')
ballAndPlayer_df = pd.read_csv(r'D:\TennisProject\player_tracking_output_13.csv')

# The annotated video from the same pipeline run (detection boxes + court
# points already drawn on it) -- shown alongside the mapping so detected
# positions can be checked directly against the source footage. Rendered
# as an image panel inside the same matplotlib figure as the 2D mapping
# (see video_ax/video_im below) rather than a separate cv2.imshow window,
# so both live in one window with the video on the left.
video_cap = cv2.VideoCapture(r'D:\TennisProject\detected_video_8.mp4')

# Downscaled display resolution -- feeding the full 1920x1080 frame into
# matplotlib's imshow every tick (needed once video moved into the same
# figure as the mapping plot) turned out to cause laggy playback: measured
# directly (matching FuncAnimation's actual blit-only redraw, not a full
# canvas draw), cost scales with pixel count -- ~192ms/blit at full
# resolution, ~41ms at 800x450, ~33ms at 640x360, ~21ms at 480x270. This is
# a direct quality-vs-smoothness dial: 640x360 is noticeably sharper than
# 480x270 but sits right at the ~33ms/frame budget for 30fps *before* the
# video-read and scatter-plot costs are even added back in, so some
# smoothness is traded back for it. Drop to 480x270 if playback feels
# laggy again, or down to 320x180 for the fastest/blurriest option.
# Resizing inside _get_video_frame's cache (not after every call) matters
# too -- otherwise a paused, repeatedly-returned frame would get re-resized
# every tick for no reason.
VIDEO_DISPLAY_W, VIDEO_DISPLAY_H = 640, 360



fps = court['fps'].iloc[0] if 'fps' in court.columns else 60
frame_time = 1/fps


COURT_WIDTH = 8.23
DOUBLES_WIDTH = 10.97
DOUBLES_ALLEY = (DOUBLES_WIDTH - COURT_WIDTH) / 2  # 1.37m each side
COURT_LENGTH = 23.77
NET_Y = COURT_LENGTH / 2
SERVICE_LINE_DIST = 6.4
NEAR_SERVICE_Y = NET_Y - SERVICE_LINE_DIST
FAR_SERVICE_Y = NET_Y + SERVICE_LINE_DIST
CENTER_X = COURT_WIDTH / 2
CENTER_MARK_LENGTH = 0.10  # ITF center mark: 4in (10cm) tick at the midpoint of each baseline


COURT_POINT_REAL_XY = {
    'BL': (0, 0), 'BR': (COURT_WIDTH, 0),
    'TL': (0, COURT_LENGTH), 'TR': (COURT_WIDTH, COURT_LENGTH),
    'NSL': (0, NEAR_SERVICE_Y), 'NSR': (COURT_WIDTH, NEAR_SERVICE_Y),
    'FSL': (0, FAR_SERVICE_Y), 'FSR': (COURT_WIDTH, FAR_SERVICE_Y),
    
}

src_points, dst_points = [], []
for name, (real_x, real_y) in COURT_POINT_REAL_XY.items():
    x_col, y_col = f'{name}_x', f'{name}_y'
    if x_col not in court.columns or y_col not in court.columns:
        continue
    px, py = court[x_col].iloc[0], court[y_col].iloc[0]
    if pd.isna(px) or pd.isna(py):
        continue
    src_points.append([px, py])
    dst_points.append([real_x, real_y])

if len(src_points) < 4:
    raise ValueError(f"Need at least 4 detected court points for a homography, found {len(src_points)}")

src_points = np.array(src_points, dtype=np.float32)
dst_points = np.array(dst_points, dtype=np.float32)
print(f"Fitting homography from {len(src_points)} detected court points")

#ทำ homography matrix จากพิกัด pixel ของ court points ในไฟล์ Excel ไปยังพิกัด meter ของ court points
M, _ = cv2.findHomography(src_points, dst_points)
print(M)

# --- Ball height correction near the net -----------------------------------
# The ground-plane homography above is only exact for points that actually
# sit on the ground (players' feet). The ball, in flight, is not -- and
# right around the net is exactly where its height is largest (it has to
# clear the net), so the homography's Z=0 assumption is most wrong exactly
# there: an elevated point looks, to a ground-plane homography, like a
# ground-level point further down the court than it really is. Verified on
# this project's own data (see project notes): a ball that was still on the
# near side transformed to a position past the net line.
#
# Freely *fitting* the ball's unknown height (a 6-parameter projectile fit
# against a single camera) was tried and rejected -- with this camera's
# achievable calibration precision (~20-30px), height and distance aren't
# well separated, and the optimizer converged to physically impossible
# trajectories (negative height, positions meters outside the court) that
# had low reprojection error but were nonsense. The fix that actually works:
# don't estimate height at all -- ASSUME it, using the one place on court
# where real height is known by rule rather than guessed: the net (0.914m
# at center, 1.07m at the posts, regulation values, interpolated linearly
# between). Near the net, intersect the ball's camera ray with that known
# height plane instead of the Z=0 ground plane; far from the net, the plain
# homography is fine (the ball is usually near the ground there anyway --
# a bounce or a pickup) and correcting it would just add camera-model noise.
#
# This still needs an actual 3D camera pose (not just the 2D homography) to
# cast a ray through a pixel -- solved via solvePnP over every court point
# with a known real-world position, crucially including the net points
# (they're NOT on the Z=0 ground plane, which is what makes a full pose
# solvable from one view at all; an all-coplanar point set is fundamentally
# ambiguous about height).
NET_CENTER_HEIGHT = 0.914  # ITF: net height at the center, meters
NET_POST_HEIGHT = 1.07     # ITF: net height at the posts (doubles sideline), meters
POST_DIST_FROM_CENTER = DOUBLES_WIDTH / 2


def _net_height_at(x):
    """Approximate net height at horizontal position x (meters), linearly
    interpolated between the center strap and the posts -- the net's real
    sag is closer to a shallow catenary, but the two curves differ by only
    a centimeter or two over this span, well under the precision anything
    else here can offer."""
    d = abs(x - CENTER_X)
    return NET_CENTER_HEIGHT + (NET_POST_HEIGHT - NET_CENTER_HEIGHT) * (d / POST_DIST_FROM_CENTER)


CALIBRATION_POINTS_XYZ = {
    'BDL': (-DOUBLES_ALLEY, 0, 0), 'BL': (0, 0, 0), 'NCM': (CENTER_X, 0, 0),
    'BR': (COURT_WIDTH, 0, 0), 'BDR': (COURT_WIDTH + DOUBLES_ALLEY, 0, 0),
    'NSL': (0, NEAR_SERVICE_Y, 0), 'NCT': (CENTER_X, NEAR_SERVICE_Y, 0), 'NSR': (COURT_WIDTH, NEAR_SERVICE_Y, 0),
    'FSL': (0, FAR_SERVICE_Y, 0), 'FCT': (CENTER_X, FAR_SERVICE_Y, 0), 'FSR': (COURT_WIDTH, FAR_SERVICE_Y, 0),
    'TDL': (-DOUBLES_ALLEY, COURT_LENGTH, 0), 'TL': (0, COURT_LENGTH, 0), 'FCM': (CENTER_X, COURT_LENGTH, 0),
    'TR': (COURT_WIDTH, COURT_LENGTH, 0), 'TDR': (COURT_WIDTH + DOUBLES_ALLEY, COURT_LENGTH, 0),
    'NetL': (-DOUBLES_ALLEY, NET_Y, NET_POST_HEIGHT),
    'NetSL': (0, NET_Y, _net_height_at(0)),
    'NetC': (CENTER_X, NET_Y, NET_CENTER_HEIGHT),
    'NetSR': (COURT_WIDTH, NET_Y, _net_height_at(COURT_WIDTH)),
    'NetR': (COURT_WIDTH + DOUBLES_ALLEY, NET_Y, NET_POST_HEIGHT),
}
NET_POINT_NAMES = {'NetL', 'NetSL', 'NetC', 'NetSR', 'NetR'}

_calib_obj, _calib_img, _calib_names = [], [], []
for _name, (_X, _Y, _Z) in CALIBRATION_POINTS_XYZ.items():
    _xcol, _ycol = f'{_name}_x', f'{_name}_y'
    if _xcol not in court.columns or _ycol not in court.columns:
        continue
    _px, _py = court[_xcol].iloc[0], court[_ycol].iloc[0]
    if pd.isna(_px) or pd.isna(_py):
        continue
    _calib_obj.append((_X, _Y, _Z))
    _calib_img.append((_px, _py))
    _calib_names.append(_name)

HEIGHT_CORRECTION_AVAILABLE = False
if len(_calib_names) >= 6 and NET_POINT_NAMES & set(_calib_names):
    _calib_obj = np.array(_calib_obj, dtype=np.float32)
    _calib_img = np.array(_calib_img, dtype=np.float32)
    _FRAME_W = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    _FRAME_H = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    # Ground points alone are exactly coplanar, so an unknown focal length
    # can't be told apart from camera pose for them (the classic single-
    # homography ambiguity) -- sweeping focal length and keeping whichever
    # minimizes reprojection error sidesteps needing a separate checkerboard
    # calibration. Verified against this project's own data to land on a
    # stable answer; letting cv2.calibrateCamera fit distortion + pixel
    # aspect ratio freely instead found a *lower* reprojection error by
    # inventing a physically-impossible lens (pixels ~6x taller than wide)
    # -- rejected for that reason despite the better-looking number.
    #
    # _CALIB_K1: a single, fixed radial-distortion term (everything else --
    # aspect ratio, principal point, tangential/higher-order distortion --
    # still locked to physical defaults). Chosen by leave-one-out cross-
    # validation on this project's own 21 calibration points (calibrate on
    # 20, measure error predicting the 21st, average over which point is
    # held out) rather than in-sample reprojection error, specifically to
    # avoid the free-distortion overfitting trap above: k1=-0.28 minimized
    # mean held-out error (33.2px -> 27.4px, ~17% tighter) with a clean,
    # single-minimum trend on either side, unlike letting distortion float
    # freely. Still a modest, not transformative, improvement -- the
    # remaining error is consistent with genuine radial lens distortion
    # (edge points fit worse than center-ish ones) that one fixed k1 can't
    # fully absorb, not a sign a further-tuned value would do much better.
    _CALIB_K1 = -0.28
    _DIST = np.array([_CALIB_K1, 0, 0, 0, 0], dtype=np.float64)
    _best = None
    for _f in np.linspace(500, 4000, 300):
        _K = np.array([[_f, 0, _FRAME_W / 2], [0, _f, _FRAME_H / 2], [0, 0, 1]], dtype=np.float64)
        _ok, _rvec, _tvec = cv2.solvePnP(_calib_obj, _calib_img, _K, _DIST, flags=cv2.SOLVEPNP_ITERATIVE)
        if not _ok:
            continue
        _proj, _ = cv2.projectPoints(_calib_obj, _rvec, _tvec, _K, _DIST)
        _err = np.sqrt(np.mean(np.sum((_proj.reshape(-1, 2) - _calib_img) ** 2, axis=1)))
        if _best is None or _err < _best[0]:
            _best = (_err, _f, _rvec, _tvec)
    _calib_err, _calib_f, _rvec, _tvec = _best
    _K = np.array([[_calib_f, 0, _FRAME_W / 2], [0, _calib_f, _FRAME_H / 2], [0, 0, 1]], dtype=np.float64)
    _R, _ = cv2.Rodrigues(_rvec)
    CAM_CENTER = (-_R.T @ _tvec).flatten()
    _R_T = _R.T
    print(f"Ball-height calibration: {len(_calib_names)} points "
          f"({', '.join(sorted(NET_POINT_NAMES & set(_calib_names)))} for net height), "
          f"RMS reprojection error {_calib_err:.1f}px")
    HEIGHT_CORRECTION_AVAILABLE = True
else:
    print("No net reference points in the court CSV -- ball position near the net will use "
          "the plain ground-plane homography (no height correction available).")


def _intersect_at_height(u, v, Z):
    """Camera ray through pixel (u,v), intersected with the horizontal
    plane at height Z -- the corrected ground position for a point known
    (or assumed) to be at that height, instead of Z=0. Undistorts first --
    the calibration's k1 describes real lens distortion, so a raw observed
    pixel has to be corrected the same way before its ray direction means
    anything relative to the (distortion-free) pose that was solved for."""
    undistorted = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), _K, _DIST).reshape(2)
    d = _R_T @ np.array([undistorted[0], undistorted[1], 1.0])
    lam = (Z - CAM_CENTER[2]) / d[2]
    return CAM_CENTER[0] + lam * d[0], CAM_CENTER[1] + lam * d[1]


NET_HEIGHT_TAPER_FULL = 1.5  # meters from net: full height-correction weight within this range
NET_HEIGHT_TAPER_ZERO = 4.0  # meters from net: correction fades out entirely by this range


def _net_proximity_weight(dist_to_net):
    # Blending the *output* position (not the assumed height) linearly
    # between the naive and height-corrected estimate keeps the corrected
    # trajectory's frame-to-frame motion in the same range as the ball's
    # actual observed speed elsewhere -- verified against this project's
    # own data: tapering the assumed height itself instead, even smoothly,
    # still produced multi-meter single-frame jumps, because ray-plane
    # intersection's sensitivity to height is itself large and nonlinear
    # here. A straight-line blend of two fixed endpoints can't do that.
    if dist_to_net <= NET_HEIGHT_TAPER_FULL:
        return 1.0
    if dist_to_net >= NET_HEIGHT_TAPER_ZERO:
        return 0.0
    return 1.0 - (dist_to_net - NET_HEIGHT_TAPER_FULL) / (NET_HEIGHT_TAPER_ZERO - NET_HEIGHT_TAPER_FULL)


#Tranform ball pixel to ball mater
meter_df = ballAndPlayer_df[['frame']].copy() #meter_df built from your actual pipeline's frame column, not the old dataset file ---

def reject_pixel_track_outliers(series, window=21):
    # Rolling (local), not whole-series, median/MAD -- a global reference
    # can't catch a track-ID switch (ByteTrack briefly locking onto a
    # different, physically-distant person or object under the same
    # track ID) if it's not rare enough in the *whole* video to look
    # anomalous overall. Verified against this project's own data: a
    # multi-frame stretch where player_1 jumped to the far player's
    # position (a ~800px, physically-impossible single-frame change)
    # didn't clear the old global-MAD threshold, because similar
    # near/far value ranges each occur often enough elsewhere in the
    # video on their own. A rolling median is dominated by whichever
    # value is locally the majority, so the same stretch reads as a
    # clear local outlier instead.
    med = series.rolling(window, center=True, min_periods=5).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=5).median()
    mad = mad.replace(0, np.nan)
    return (0.6745 * (series - med) / mad).fillna(0)



_FRAME_HEIGHT_PX = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

for prefix in ['player_1', 'far_player']:
    x_col, y_col = f'{prefix}_x', f'{prefix}_y'
    if x_col not in ballAndPlayer_df.columns:
        continue
    # A tracked box whose bottom sits exactly at the frame edge isn't a
    # real foot position -- it's the box clipped by the frame boundary
    # (the player's actual feet are below the visible frame, which happens
    # when they're right up against the near baseline, close to the
    # camera). Verified against this project's own data: several
    # consecutive frames at a real hit moment all reported foot_y ==
    # frame height exactly, and the near-camera homography (already known
    # to be highly sensitive there -- see OUTLIER_MARGIN's note) turned
    # that into a player position ~2.5m behind the baseline, an
    # implausible spot for that shot. This was the actual cause of
    # "player and ball look together in the video but are far apart on
    # the map" at contact moments close to the camera -- not a ball-side
    # issue at all, a player-tracking one.
    clipped = ballAndPlayer_df[y_col] >= _FRAME_HEIGHT_PX - 1
    combined_z = np.hypot(reject_pixel_track_outliers(ballAndPlayer_df[x_col]),
                           reject_pixel_track_outliers(ballAndPlayer_df[y_col]))
    outlier = ballAndPlayer_df[x_col].notna() & (combined_z > 4.0)
    ballAndPlayer_df.loc[outlier | clipped, [x_col, y_col]] = np.nan

def _smooth_interpolate(series, lo=None, hi=None):
    # Linear, not pchip: a cubic fit through sparse/irregular gaps (long runs
    # of consecutive missing/rejected frames leave few real points to fit
    # through) can overshoot far past its neighboring points -- measured on
    # this project's own data as single-*frame* jumps of 20-30+ meters,
    # which alone inflated the reported max ball speed to 1151 m/s. The
    # final .clip() bounds the absolute value but doesn't stop that kind of
    # spike, since both the spike and the clipped result can individually
    # sit inside [lo, hi]. A straight line between two valid points can
    # never overshoot past them, which removes the failure mode outright --
    # the tradeoff is gaps get bridged as straight segments rather than a
    # curve, a minor cosmetic loss next to fabricating a 30-meter phantom jump.
    filled = series.interpolate(method='linear', limit_direction='both')
    if lo is not None or hi is not None:
        filled = filled.clip(lower=lo, upper=hi)
    return filled


# Captured before interpolation fills it in -- the full-flight correction
# needs to know exactly which frames had no real ball detection at all, not
# just which ones look implausible after a pixel-space straight line has
# already been drawn through the gap.
_ball_raw_missing = ballAndPlayer_df['ball_x'].isna().to_numpy()

for col in ['ball_x', 'ball_y', 'player_1_x', 'player_1_y', 'far_player_x', 'far_player_y']:
    if col in ballAndPlayer_df.columns:
        ballAndPlayer_df[col] = _smooth_interpolate(ballAndPlayer_df[col])



OUTLIER_MARGIN = 5.0  # meters beyond the court rectangle still treated as plausible


# pchip interpolation (in _smooth_interpolate) assumes the implausible frames
# it's bridging are a minority -- occasional misdetections surrounded by good
# data on both sides, which a cubic fit through the surrounding real points
# can sensibly curve across. That assumption breaks down for a track that's
# implausible MOST of the time (seen with a far-court point sitting close to
# the camera's vanishing point, where the homography is so sensitive that
# even correct, stable pixel detections transform to positions well outside
# the court): there's no real signal left to interpolate through, so pchip
# ends up curve-fitting mostly-missing data and swinging wildly between
# frames, which then gets clipped at whichever boundary it overshot -- the
# clip bounds it, but the frame-to-frame result still looks like it's
# teleporting between the near and far clip walls. Past this threshold,
# skip interpolation and clip the raw transform directly instead: not more
# accurate (the underlying position estimate is still unreliable that far
# out), but honest and stable rather than fabricating false motion.
IMPLAUSIBLE_FRACTION_FOR_DIRECT_CLIP = 0.3


def transform_to_meters(x_col, y_col, correct_net_height=False):
    pixel_points = ballAndPlayer_df[[x_col, y_col]].to_numpy(dtype=np.float32).reshape(-1, 1, 2)
    meter_points = cv2.perspectiveTransform(pixel_points, M).reshape(-1, 2)
    mx, my = meter_points[:, 0].copy(), meter_points[:, 1].copy()

    if correct_net_height and HEIGHT_CORRECTION_AVAILABLE:
        px_all = ballAndPlayer_df[x_col].to_numpy(dtype=np.float64)
        py_all = ballAndPlayer_df[y_col].to_numpy(dtype=np.float64)
        for i in range(len(mx)):
            if np.isnan(px_all[i]):
                continue
            w = _net_proximity_weight(abs(my[i] - NET_Y))
            if w <= 0:
                continue
            z = _net_height_at(np.clip(mx[i], 0, COURT_WIDTH))
            hx, hy = _intersect_at_height(px_all[i], py_all[i], z)
            mx[i] = mx[i] * (1 - w) + hx * w
            my[i] = my[i] * (1 - w) + hy * w

    x, y = pd.Series(mx), pd.Series(my)
    implausible = ((x < -OUTLIER_MARGIN) | (x > COURT_WIDTH + OUTLIER_MARGIN) |
                   (y < -OUTLIER_MARGIN) | (y > COURT_LENGTH + OUTLIER_MARGIN))

    if implausible.mean() > IMPLAUSIBLE_FRACTION_FOR_DIRECT_CLIP:
        x_filled = x.clip(lower=-OUTLIER_MARGIN, upper=COURT_WIDTH + OUTLIER_MARGIN)
        y_filled = y.clip(lower=-OUTLIER_MARGIN, upper=COURT_LENGTH + OUTLIER_MARGIN)
        return x_filled.to_numpy(), y_filled.to_numpy(), implausible.to_numpy()

    x[implausible] = np.nan
    y[implausible] = np.nan
    x_filled = _smooth_interpolate(x, lo=-OUTLIER_MARGIN, hi=COURT_WIDTH + OUTLIER_MARGIN)
    y_filled = _smooth_interpolate(y, lo=-OUTLIER_MARGIN, hi=COURT_LENGTH + OUTLIER_MARGIN)
    return x_filled.to_numpy(), y_filled.to_numpy(), implausible.to_numpy()


meter_df['ball_meter_x'], meter_df['ball_meter_y'], _ball_implausible = transform_to_meters(
    'ball_x', 'ball_y', correct_net_height=True)
meter_df['player_1_x'], meter_df['player_1_y'], _ = transform_to_meters('player_1_x', 'player_1_y')
# "player_2" downstream (plots, legends, distances) is sourced from
# far_player_x/y -- see the note above the pixel-outlier pass.
meter_df['player_2_x'], meter_df['player_2_y'], _ = transform_to_meters('far_player_x', 'far_player_y')


# --- Full-flight straight-line correction -----------------------------------
# The net-height fix above only anchors a height assumption at one specific,
# known location (the net) -- there's no equivalent "standard height" to
# assume anywhere else, so the same per-frame ray-cast trick can't just be
# repeated everywhere else the ball is airborne. But a real ball'sTOP-DOWN
# (X,Y) path is a straight line between whatever events change its velocity
# -- a hit or a bounce -- because gravity only curves the *vertical*
# component; nothing curves the horizontal one. So rather than estimating
# height at every interior frame (the ill-conditioned 6-parameter fit tried
# earlier, rejected for converging to physically-impossible trajectories),
# each flight segment's two endpoints -- already the most trustworthy points
# in it, and already net-height-corrected above if one happens to land near
# the net -- are connected with a straight line, replacing the wobble a
# per-frame Z=0 homography produces for every interior frame where the ball
# is actually elevated, not just the ones near the net.
#
# Segment boundaries (hit/bounce events) are found as direction reversals in
# the ball's own (already net-corrected) down-court position: a hit clearly
# reverses which end of the court the ball is heading toward, which shows up
# cleanly as a local min/max in ball_meter_y. A bounce that doesn't change
# the ball's overall direction (the common case: it keeps travelling the
# same way, just off the ground) won't show up this way and so isn't
# detected as its own boundary -- an accepted simplification, since a real
# bounce mostly preserves horizontal velocity anyway (tennis balls lose
# little horizontal speed on a bounce compared to the vertical rebound), so
# a straight line drawn straight through an undetected bounce is still a
# good approximation of the true path.
def _refine_extremum(y_values, idx, mode, window=5):
    # find_peaks runs on a 5-frame-smoothed signal (needed so noise doesn't
    # register as its own tiny reversal) -- but smoothing a shallow, gradual
    # approach to a minimum/maximum can shift *where* it looks like the
    # extremum falls by a couple of frames. Verified against this project's
    # own data at a real hit: the true minimum (ball still descending
    # toward the racket) was at frame 271, but the smoothed signal peaked
    # 2 frames early at 269 -- close enough to not look wrong in isolation,
    # but far enough that the straight line drawn from the wrong frame put
    # the ball meters from the player at the exact moment of contact. This
    # snaps each candidate to the true local extremum in the *raw* signal
    # within a small window, so the boundary lands on the real event frame
    # instead of wherever smoothing happened to place it.
    lo, hi = max(0, idx - window), min(len(y_values), idx + window + 1)
    local = y_values[lo:hi]
    return lo + (int(np.argmin(local)) if mode == 'min' else int(np.argmax(local)))


def _segment_ball_flight(y_values, trusted_mask, min_separation_frames=8, min_prominence=0.5):
    smoothed = pd.Series(y_values).rolling(5, center=True, min_periods=1).mean().to_numpy()
    peaks, _ = find_peaks(smoothed, distance=min_separation_frames, prominence=min_prominence)
    troughs, _ = find_peaks(-smoothed, distance=min_separation_frames, prominence=min_prominence)
    peaks = [_refine_extremum(y_values, i, 'max') for i in peaks]
    troughs = [_refine_extremum(y_values, i, 'min') for i in troughs]
    # A "peak"/"trough" sitting on an untrustworthy frame isn't a real
    # hit/bounce -- verified against this project's own data: a third of
    # naively-detected boundaries landed within 1m of OUTLIER_MARGIN's
    # clip ceiling, a region already known (see the homography/vanishing-
    # point note above OUTLIER_MARGIN) to amplify small pixel noise into
    # multi-meter swings that only coincidentally look like a direction
    # reversal. Filtered out here rather than trusted as real events.
    events = [i for i in (*peaks, *troughs) if trusted_mask[i]]
    return sorted({0, len(y_values) - 1} | set(events))


# Real hit/bounce locations don't land meters past the baseline -- a
# tighter bound than OUTLIER_MARGIN (which exists to decide "keep this
# frame at all" for the broader trajectory, not "trust this frame enough
# to anchor a whole straight-line segment on it").
# Only Y is checked against the tighter margin -- Y (down-court distance)
# is both the axis segmentation actually runs on and the one confirmed,
# against this project's own data, to be where far-court homography noise
# clusters. X was tried too at first, but that rejected a genuine, large
# far-court reversal (a real wide shot, ball_x roughly 3m past the
# sideline) just because its X fell outside the same tight margin --
# collapsing two real rally exchanges into one wrong straight line.
# ball_implausible (the existing, more generous +-OUTLIER_MARGIN box on
# both axes) still catches genuinely broken X values.
_SEGMENT_TRUST_MARGIN = 1.5
_ball_y_arr = meter_df['ball_meter_y'].to_numpy()
_trusted_for_segmentation = (
    ~_ball_implausible
    & (_ball_y_arr > -_SEGMENT_TRUST_MARGIN) & (_ball_y_arr < COURT_LENGTH + _SEGMENT_TRUST_MARGIN)
)
_flight_boundaries = _segment_ball_flight(meter_df['ball_meter_y'].to_numpy(), _trusted_for_segmentation)


def _gap_boundaries(raw_missing, min_gap_frames=5):
    """Frames bracketing any real-detection gap long enough that a straight
    pixel-space interpolation through it (see _smooth_interpolate) stops
    being a reasonable stand-in for "we don't know". Verified against this
    project's own data: the worst gaps run up to 53 frames (~1.8s), and
    several sit partly in the far-court region where tiny pixel drift
    explodes into a huge apparent swing -- the interpolated bridge doesn't
    just guess wrong, it can look enough like a real hit/bounce to fool the
    peak-based detection above (a frozen-then-slowly-climbing interpolation
    artifact was found masquerading as a legitimate direction reversal), or
    just sit un-anchored as an invented multi-second arc if it doesn't.
    Forcing a boundary at the last real detection before the gap and the
    first one after it guarantees the straight-line correction always
    connects two genuinely observed points across a gap, never trusts
    whatever the raw interpolation invented in between."""
    n = len(raw_missing)
    boundaries = []
    i = 0
    while i < n:
        if raw_missing[i]:
            j = i
            while j < n and raw_missing[j]:
                j += 1
            if j - i >= min_gap_frames:
                if i > 0:
                    boundaries.append(i - 1)
                if j < n:
                    boundaries.append(j)
            i = j
        else:
            i += 1
    return boundaries


_gap_forced_boundaries = _gap_boundaries(_ball_raw_missing, min_gap_frames=5)
_flight_boundaries = sorted(set(_flight_boundaries) | set(_gap_forced_boundaries))
_ball_x = meter_df['ball_meter_x'].to_numpy().copy()
_ball_y = meter_df['ball_meter_y'].to_numpy().copy()
for _b0, _b1 in zip(_flight_boundaries[:-1], _flight_boundaries[1:]):
    _n = _b1 - _b0
    if _n < 2:
        continue
    _t = np.linspace(0.0, 1.0, _n + 1)
    _ball_x[_b0:_b1 + 1] = _ball_x[_b0] + _t * (_ball_x[_b1] - _ball_x[_b0])
    _ball_y[_b0:_b1 + 1] = _ball_y[_b0] + _t * (_ball_y[_b1] - _ball_y[_b0])
meter_df['ball_meter_x'] = _ball_x
meter_df['ball_meter_y'] = _ball_y
print(f"Full-flight correction: {len(_flight_boundaries) - 1} straight-line segments "
      f"({len(_flight_boundaries)} hit/bounce-like boundary frames detected)")


SMOOTHING_WINDOW = 45
for prefix in ['player_1', 'player_2']:
    x_col, y_col = f'{prefix}_x', f'{prefix}_y'
    meter_df[x_col] = meter_df[x_col].rolling(SMOOTHING_WINDOW, center=True, min_periods=1).mean()
    meter_df[y_col] = meter_df[y_col].rolling(SMOOTHING_WINDOW, center=True, min_periods=1).mean()

# --- compute ball speed and store in meter_df (not the undefined ball_df: Mae ni wa using ball_df) ---
#From now on will be meter
#Compute ball Speed
dx_court = np.diff(meter_df['ball_meter_x'])
dy_court = np.diff(meter_df['ball_meter_y'])
distance_meter = np.sqrt(dx_court**2 + dy_court**2)
speed_meter_per_sec = distance_meter / frame_time
speed_meter_per_sec = np.insert(speed_meter_per_sec,0, np.nan)
meter_df['ball_speed_m_per_sec'] = speed_meter_per_sec   # FIX: consistent name used later in the plot

print(meter_df['ball_speed_m_per_sec'].head())
avg_ball_speed = meter_df['ball_speed_m_per_sec'].mean()
max_ball_speed = meter_df['ball_speed_m_per_sec'].max()

# --- Distance from each player to the ball ---
meter_df['player_1_to_ball_distance'] = np.sqrt(
    (meter_df['player_1_x'] - meter_df['ball_meter_x'])**2 +
    (meter_df['player_1_y'] - meter_df['ball_meter_y'])**2
)
meter_df['player_2_to_ball_distance'] = np.sqrt(
    (meter_df['player_2_x'] - meter_df['ball_meter_x'])**2 +
    (meter_df['player_2_y'] - meter_df['ball_meter_y'])**2
)

print("Meter DataFrame")
print(meter_df[['ball_meter_x', 'ball_meter_y', 'player_1_x', 'player_1_y', 'player_2_x',
                'player_2_y', 'player_1_to_ball_distance', 'player_2_to_ball_distance']].head())


#Picking up frame to map.
#Not doing the whole video yet ----> DOING ---> Done
#frame_row = meter_df.iloc[686]

#Giving the court scale

#The video 2DMapping pipline from  the top view of the court, the ball and the player

total_frames = len(meter_df)
total_time = total_frames / fps
trail_length = 0


# The video panel and the mapping plot are both driven from the same
# `update()` callback (it re-seeks video_cap to whatever frame it's about to
# draw, every tick, rather than reading frames sequentially) -- so a shared
# playback position naturally keeps both in sync; reverse/seek/speed just
# need to control what that shared position does between ticks.
class Playback:
    def __init__(self, total):
        self.total = total
        self.pos = 0.0     # float so a fractional speed (e.g. 0.5x) still accumulates correctly
        self.speed = 1.0
        self.direction = 1  # 1 = forward, -1 = reverse
        self.paused = False

    def advance(self):
        if self.paused:
            return
        self.pos += self.speed * self.direction
        if self.pos <= 0.0:
            self.pos = 0.0
            self.paused = True   # stop cleanly at either end instead of trying to run past it
        elif self.pos >= self.total - 1:
            self.pos = self.total - 1
            self.paused = True

    def seek(self, delta_frames):
        self.pos = max(0.0, min(self.total - 1, self.pos + delta_frames))

    @property
    def frame_idx(self):
        return int(round(self.pos))


playback = Playback(total_frames)
SPEED_MIN, SPEED_MAX, SPEED_STEP = 0.1, 8.0, 1.5
SEEK_SECONDS = 1.0


def handle_key(key):
    """'left'/'right' match matplotlib's own event.key strings for the arrow
    keys."""
    if key == ' ':
        playback.paused = not playback.paused
    elif key == 'w':
        playback.speed = min(SPEED_MAX, playback.speed * SPEED_STEP)
    elif key == 'x':
        playback.speed = max(SPEED_MIN, playback.speed / SPEED_STEP)
    elif key == 'left':
        playback.seek(-fps * SEEK_SECONDS)
    elif key == 'right':
        playback.seek(fps * SEEK_SECONDS)
    elif key == 'z':
        playback.direction *= -1


# matplotlib binds left/right by default to its toolbar's back/forward view
# history -- harmless here (there's no zoom/pan history to navigate) but it
# would silently compete with using them for seeking, so those bindings are
# removed rather than left to coexist.
plt.rcParams['keymap.back'] = [k for k in plt.rcParams['keymap.back'] if k != 'left']
plt.rcParams['keymap.forward'] = [k for k in plt.rcParams['keymap.forward'] if k != 'right']

print("Playback controls (click the window first so it has keyboard focus):")
print("  space = play/pause   w/x = speed up/down   left/right arrows = seek back/forward 1s   z = reverse direction")


def draw_court(ax, zorder=1, fill_color=None):
    
    left_d, right_d = -DOUBLES_ALLEY, COURT_WIDTH + DOUBLES_ALLEY
    line_color = '#E8EDF2'
    if fill_color:
        ax.fill([left_d, right_d, right_d, left_d], [0, 0, COURT_LENGTH, COURT_LENGTH],
                color=fill_color, zorder=zorder - 1)
    # outer boundary: baseline + doubles sidelines
    ax.plot([left_d, right_d, right_d, left_d, left_d], [0, 0, COURT_LENGTH, COURT_LENGTH, 0],
            color=line_color, linewidth=1, zorder=zorder)
    # singles sidelines, inset from the doubles sidelines
    ax.plot([0, 0], [0, COURT_LENGTH], color=line_color, linewidth=1, zorder=zorder)
    ax.plot([COURT_WIDTH, COURT_WIDTH], [0, COURT_LENGTH], color=line_color, linewidth=1, zorder=zorder)
    ax.plot([0, COURT_WIDTH], [NEAR_SERVICE_Y, NEAR_SERVICE_Y], color=line_color, linewidth=1, zorder=zorder)
    ax.plot([0, COURT_WIDTH], [FAR_SERVICE_Y, FAR_SERVICE_Y], color=line_color, linewidth=1, zorder=zorder)
    ax.plot([CENTER_X, CENTER_X], [NEAR_SERVICE_Y, FAR_SERVICE_Y], color=line_color, linewidth=1, zorder=zorder)
    ax.plot([left_d, right_d], [NET_Y, NET_Y], color='dimgray', linewidth=2, zorder=zorder)
    ax.plot([CENTER_X, CENTER_X], [0, CENTER_MARK_LENGTH], color=line_color, linewidth=1, zorder=zorder)
    ax.plot([CENTER_X, CENTER_X], [COURT_LENGTH - CENTER_MARK_LENGTH, COURT_LENGTH],
            color=line_color, linewidth=1, zorder=zorder)



fig = plt.figure(figsize=(13, 7.5))
grid = fig.add_gridspec(1, 3, width_ratios=[5, 2.4, 1.1], wspace=0.08)
video_ax = fig.add_subplot(grid[0, 0])
mapping_window = fig.add_subplot(grid[0, 1])
info_ax = fig.add_subplot(grid[0, 2])
fig.canvas.manager.set_window_title('Tennis Tracking')

video_ax.axis('off')
video_ax.set_title('Detected Video')
# interpolation='bilinear': smoother-looking scaling than the default
# nearest-neighbor blockiness when the panel renders larger than the
# downscaled frame -- measured at under 1ms/frame more than 'nearest' at
# this resolution, cheap enough to just take the better look.
video_im = video_ax.imshow(np.zeros((VIDEO_DISPLAY_H, VIDEO_DISPLAY_W, 3), dtype=np.uint8),
                            interpolation='bilinear')


draw_court(mapping_window, fill_color='#133458')

# Matches OUTLIER_MARGIN, not an independent margin -- a position that
# transform_to_meters accepts as plausible (within OUTLIER_MARGIN of the
# court) must actually be visible here, or it's real data that silently
# never appears on screen. A tighter margin here previously left ~23% of
# genuinely valid ball positions plotted off-axis, invisible despite being
# in meter_df with correct values.
mapping_window.set_xlim(-OUTLIER_MARGIN, COURT_WIDTH + OUTLIER_MARGIN)
mapping_window.set_ylim(-OUTLIER_MARGIN, COURT_LENGTH + OUTLIER_MARGIN)
mapping_window.set_aspect('equal')
mapping_window.set_title('2D Mapping Window')

ball_scatter = mapping_window.scatter([], [], color='#838921', s=40)
player1_scatter = mapping_window.scatter([], [], color='#BD4444', marker='o', s=60)
player2_scatter = mapping_window.scatter([], [], color='#BD4444', marker='o', s=60)

info_ax.axis('off')
info_ax.set_xlim(0, 1)
info_ax.set_ylim(0, 1)
info_ax.scatter([], [], color='#838921', marker='o', label='Ball')
info_ax.scatter([], [], color='#BD4444', marker='o', label='Player 1')
info_ax.scatter([], [], color='#BD4444', marker='^', label='Player 2')
info_ax.legend(loc='center', bbox_to_anchor=(0.5, 0.35), frameon=False, fontsize=9)

# NOTE: this has to live in an axes as a normal text artist (not
# ax.set_title(), which sits just above its axes' bbox and so never actually
# redraws once blit=True is on -- see previous fix). info_ax, not
# mapping_window, so it can never land on top of / behind a player or the ball.
title_text = info_ax.text(0.5, 0.95, '', ha='center', va='top', fontsize=9, wrap=True)


def _trail_facecolors(hex_color, n):
    r, g, b = to_rgb(hex_color)
    alphas = np.arange(1, n + 1) / n if n else np.empty(0)
    return np.column_stack([np.full(n, r), np.full(n, g), np.full(n, b), alphas])


# video_cap.set(CAP_PROP_POS_FRAMES) before every read (needed to support
# reverse/seek at all) turned out to be the actual reason "1x" played back
# slower than real time -- measured directly on this project's own output
# video: seek+read costs ~38ms/frame, most of a 30fps frame's entire 33ms
# budget, before matplotlib's own redraw is even added. A plain sequential
# .read() (no seek) measured at ~6ms/frame instead. Forward playback at any
# normal speed only ever needs the next few frames in decode order, so it
# can skip the expensive seek and just read (and discard) its way there;
# only an actual jump -- seeking, reversing, or resuming after a pause that
# moved the position -- still needs the real (slow) seek.
_video_cache = {"frame_idx": None, "image": None}
SEQUENTIAL_READ_MAX_SKIP = 30  # ~1s at 30fps -- beyond this a seek is cheaper than reading through it


def _get_video_frame(frame_idx):
    if _video_cache["frame_idx"] == frame_idx and _video_cache["image"] is not None:
        return _video_cache["image"]  # paused / same frame as last tick -- nothing to decode

    last = _video_cache["frame_idx"]
    if last is not None and 0 <= frame_idx - last <= SEQUENTIAL_READ_MAX_SKIP:
        for _ in range(frame_idx - last - 1):
            video_cap.read()
        ok, img = video_cap.read()
    else:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = video_cap.read()

    if not ok:
        return None
    img = cv2.resize(img, (VIDEO_DISPLAY_W, VIDEO_DISPLAY_H), interpolation=cv2.INTER_AREA)
    _video_cache["frame_idx"] = frame_idx
    _video_cache["image"] = img
    return img


def update(_):
    playback.advance()
    frame_idx = playback.frame_idx

    start = max(0, frame_idx - trail_length)
    trail = meter_df.iloc[start:frame_idx + 1]
    n = len(trail)

    ball_scatter.set_offsets(trail[['ball_meter_x', 'ball_meter_y']].to_numpy())
    ball_scatter.set_facecolor(_trail_facecolors('#838921', n))

    player1_scatter.set_offsets(trail[['player_1_x', 'player_1_y']].to_numpy())
    player1_scatter.set_facecolor(_trail_facecolors('#BD4444', n))

    player2_scatter.set_offsets(trail[['player_2_x', 'player_2_y']].to_numpy())
    player2_scatter.set_facecolor(_trail_facecolors('#BD4444', n))

    current_time = frame_idx / fps
    state_label = 'Paused' if playback.paused else ('Reverse' if playback.direction < 0 else 'Play')
    direction_arrow = '<' if playback.direction < 0 else '>'
    title_text.set_text(
        f'Frame {frame_idx} / {total_frames}\nTime: {current_time:.2f}s / {total_time:.2f}s\n'
        f'{state_label} {direction_arrow} {playback.speed:.2g}x'
    )

    video_frame = _get_video_frame(frame_idx)
    if video_frame is not None:
        video_im.set_data(video_frame[:, :, ::-1])  # BGR (cv2) -> RGB (matplotlib)

    return ball_scatter, player1_scatter, player2_scatter, title_text, video_im


def on_mpl_key(event):
    handle_key(event.key)


fig.canvas.mpl_connect('key_press_event', on_mpl_key)


#Plot seaborn for the ball speed

fig2, information_window = plt.subplots(2,2,figsize=(6, 8.5))
fig2.canvas.manager.set_window_title('Information Window')

#วาดกรอบ
draw_court(information_window[0, 0], zorder=1)
# scatterplot colored by distance to ball
sns.scatterplot(
    ax = information_window[0,0],
    data=meter_df,
    x='player_1_x',
    y='player_1_y',
    hue='player_1_to_ball_distance',
    palette='viridis',
    s=60,
    zorder=2
)

information_window[0,0].set_xlim(-OUTLIER_MARGIN, COURT_WIDTH + OUTLIER_MARGIN)
information_window[0,0].set_ylim(-OUTLIER_MARGIN, COURT_LENGTH + OUTLIER_MARGIN)
information_window[0,0].set_aspect('equal')
information_window[0,0].set_title('Player 1 - Distance to Ball')
information_window[0,0].set_xlabel('Court width (m)')
information_window[0,0].set_ylabel('Court length (m)')


draw_court(information_window[0, 1], zorder=1)
# scatterplot colored by distance to ball
sns.scatterplot(
    ax = information_window[0,1],
    data=meter_df,
    x='player_2_x',
    y='player_2_y',
    hue='player_2_to_ball_distance',
    palette='viridis',
    s=60,
    zorder=2
)

information_window[0,1].set_xlim(-OUTLIER_MARGIN, COURT_WIDTH + OUTLIER_MARGIN)
information_window[0,1].set_ylim(-OUTLIER_MARGIN, COURT_LENGTH + OUTLIER_MARGIN)
information_window[0,1].set_aspect('equal')
information_window[0,1].set_title('Player 2 - Distance to Ball')
information_window[0,1].set_xlabel('Court width (m)')
information_window[0,1].set_ylabel('Court length (m)')


#Ball Speed showed

peak, _ =find_peaks(-meter_df['player_1_to_ball_distance'], distance=10)
hit_frame_indices = meter_df.iloc[peak].index

sns.lineplot(
    data=meter_df, 
    x=meter_df.index, 
    y='ball_speed_m_per_sec', 
    color='steelblue',
    ax = information_window[1,0])

for idx in hit_frame_indices:
    information_window[1, 0].axvline(x=idx, color='red', linestyle='--', alpha=0.6)

information_window[1, 0].set_xlabel('Frame')
information_window[1, 0].set_ylabel('Ball speed (m/s)')
information_window[1, 0].set_title('Ball speed, with candidate hit frames')

# frames=itertools.count() instead of range(0, total_frames): the frame to
# render now comes from `playback` (which speed/direction/seeking mutate),
# not from the value FuncAnimation passes to update() -- an unbounded
# counter just keeps ticking at a fixed wall-clock rate for as long as the
# window stays open, however far/whichever way playback actually moves.
ani = animation.FuncAnimation(fig, update, frames=itertools.count(), interval=1000/fps,
                               blit=True, cache_frame_data=False)

plt.tight_layout()
plt.show()

video_cap.release()