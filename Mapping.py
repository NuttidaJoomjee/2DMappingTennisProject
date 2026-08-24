import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_rgb
import seaborn as sns
from scipy.signal import find_peaks
import cv2

court = pd.read_csv(r'D:\TennisProject\court_coordinates_10.csv')
ballAndPlayer_df = pd.read_csv(r'D:\TennisProject\player_tracking_output_8.csv')

# The annotated video from the same pipeline run (detection boxes + court
# points already drawn on it) -- shown alongside the mapping so detected
# positions can be checked directly against the source footage.
video_cap = cv2.VideoCapture(r'D:\TennisProject\detected_video_10.mp4')



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
    'NCT': (CENTER_X, NEAR_SERVICE_Y),
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

#Tranform ball pixel to ball mater
meter_df = ballAndPlayer_df[['frame']].copy() #meter_df built from your actual pipeline's frame column, not the old dataset file ---

def reject_pixel_track_outliers(series, threshold=4.0):
    med = series.median()
    mad = (series - med).abs().median()
    if not mad or np.isnan(mad):
        return series
    return 0.6745 * (series - med) / mad



for prefix in ['player_1', 'far_player']:
    x_col, y_col = f'{prefix}_x', f'{prefix}_y'
    if x_col not in ballAndPlayer_df.columns:
        continue
    combined_z = np.hypot(reject_pixel_track_outliers(ballAndPlayer_df[x_col]),
                           reject_pixel_track_outliers(ballAndPlayer_df[y_col]))
    outlier = ballAndPlayer_df[x_col].notna() & (combined_z > 4.0)
    ballAndPlayer_df.loc[outlier, [x_col, y_col]] = np.nan

def _smooth_interpolate(series, lo=None, hi=None):
    """Fill gaps with a shape-preserving cubic fit through all the valid
    points around a gap (pchip), not just a straight line kinked between
    the two points nearest the gap's edges -- a real shot's track curves
    through a gap rather than snapping between straight segments. Falls
    back to linear when there aren't enough surrounding points for a cubic
    fit (e.g. a gap at the very start/end of the clip).

    pchip avoids *local* overshoot between adjacent knots, but on data
    this sparse and irregularly gapped (long stretches of consecutive
    implausible frames -- see OUTLIER_MARGIN below -- leave very few real
    knots to fit through), it can still swing wildly beyond any sane range
    over a whole gap -- measured on this project's own data, y values as
    extreme as -884m on a 23.77m court. So when the caller knows a
    plausible range (lo/hi), clip to it as a hard safety net; pixel-space
    columns (no lo/hi given) skip this, since there's no cheap equivalent
    bound to check them against.
    """
    try:
        filled = series.interpolate(method='pchip', limit_direction='both')
    except (ValueError, TypeError):
        filled = series.interpolate(limit_direction='both')
    if lo is not None or hi is not None:
        filled = filled.clip(lower=lo, upper=hi)
    return filled


for col in ['ball_x', 'ball_y', 'player_1_x', 'player_1_y', 'far_player_x', 'far_player_y']:
    if col in ballAndPlayer_df.columns:
        ballAndPlayer_df[col] = _smooth_interpolate(ballAndPlayer_df[col])



OUTLIER_MARGIN = 5.0  # meters beyond the court rectangle still treated as plausible


def transform_to_meters(x_col, y_col):
    pixel_points = ballAndPlayer_df[[x_col, y_col]].to_numpy(dtype=np.float32).reshape(-1, 1, 2)
    meter_points = cv2.perspectiveTransform(pixel_points, M).reshape(-1, 2)
    x, y = pd.Series(meter_points[:, 0]), pd.Series(meter_points[:, 1])
    implausible = ((x < -OUTLIER_MARGIN) | (x > COURT_WIDTH + OUTLIER_MARGIN) |
                   (y < -OUTLIER_MARGIN) | (y > COURT_LENGTH + OUTLIER_MARGIN))
    x[implausible] = np.nan
    y[implausible] = np.nan
    x_filled = _smooth_interpolate(x, lo=-OUTLIER_MARGIN, hi=COURT_WIDTH + OUTLIER_MARGIN)
    y_filled = _smooth_interpolate(y, lo=-OUTLIER_MARGIN, hi=COURT_LENGTH + OUTLIER_MARGIN)
    return x_filled.to_numpy(), y_filled.to_numpy()


meter_df['ball_meter_x'], meter_df['ball_meter_y'] = transform_to_meters('ball_x', 'ball_y')
meter_df['player_1_x'], meter_df['player_1_y'] = transform_to_meters('player_1_x', 'player_1_y')
# "player_2" downstream (plots, legends, distances) is sourced from
# far_player_x/y -- see the note above the pixel-outlier pass.
meter_df['player_2_x'], meter_df['player_2_y'] = transform_to_meters('far_player_x', 'far_player_y')


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
trail_length = 10


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



fig = plt.figure(figsize=(6, 8))
grid = fig.add_gridspec(1, 2, width_ratios=[4, 1.3], wspace=0.05)
mapping_window = fig.add_subplot(grid[0, 0])
info_ax = fig.add_subplot(grid[0, 1])
fig.canvas.manager.set_window_title('2D Mapping Window')


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


def update(frame_idx):
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
    title_text.set_text(f'Frame {frame_idx} / {total_frames}\nTime: {current_time:.2f}s\n/ {total_time:.2f}s')

    
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    video_read_ok, video_frame = video_cap.read()
    if video_read_ok:
        cv2.imshow('Video', video_frame)
        cv2.waitKey(1)

    return ball_scatter, player1_scatter, player2_scatter, title_text


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

ani = animation.FuncAnimation(fig, update, frames=range(0, total_frames), interval=1000/fps,
                               blit=True, repeat=False)

plt.tight_layout()
plt.show()

video_cap.release()
cv2.destroyAllWindows()