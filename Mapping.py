import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_rgb
import seaborn as sns
from scipy.signal import find_peaks
import cv2

court = pd.read_csv(r'D:\TennisProject\court_coordinates_3.csv')
ballAndPlayer_df = pd.read_csv(r'D:\TennisProject\player_tracking_output_3.csv')

# The annotated video from the same pipeline run (detection boxes + court
# points already drawn on it) -- shown alongside the mapping so detected
# positions can be checked directly against the source footage.
video_cap = cv2.VideoCapture(r'D:\TennisProject\detected_video_3.mp4')


# Read the source video's actual fps so playback speed and speed calculations
# match it exactly, instead of assuming 60 (older court CSVs without this
# column fall back to 60).
fps = court['fps'].iloc[0] if 'fps' in court.columns else 60
frame_time = 1/fps

# Standard ITF singles court, meters
COURT_WIDTH = 8.23
COURT_LENGTH = 23.77
NET_Y = COURT_LENGTH / 2
SERVICE_LINE_DIST = 6.4
NEAR_SERVICE_Y = NET_Y - SERVICE_LINE_DIST
FAR_SERVICE_Y = NET_Y + SERVICE_LINE_DIST
CENTER_X = COURT_WIDTH / 2
CENTER_MARK_LENGTH = 0.10  # ITF center mark: 4in (10cm) tick at the midpoint of each baseline

# Real-world (meters) position of every point court_coordinates_*.csv may contain.
# Near = smaller y, far = bigger y. Not every point is always detected (e.g. NCT
# is best-effort), so build the homography from whichever columns are present.
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

# findHomography (least-squares over all points) instead of getPerspectiveTransform
# (which only ever takes exactly 4) -- with service-line/center-line points included
# alongside the 4 corners, this averages out noise in any single detected line.
M, _ = cv2.findHomography(src_points, dst_points)
print(M)

#Tranform ball pixel to ball mater
meter_df = ballAndPlayer_df[['frame']].copy() #meter_df built from your actual pipeline's frame column, not the old dataset file ---

for col in ['ball_x', 'ball_y', 'player_1_x', 'player_1_y', 'player_2_x', 'player_2_y']:
    if col in ballAndPlayer_df.columns:
        ballAndPlayer_df[col] = ballAndPlayer_df[col].interpolate()


# --- Transform ball pixel -> meter ---
points = ballAndPlayer_df[['ball_x', 'ball_y']].to_numpy(dtype=np.float32)
points_reshaped = points.reshape(-1, 1, 2)
court_points = cv2.perspectiveTransform(points_reshaped, M)
court_points = court_points.reshape(-1, 2)
meter_df['ball_meter_x'] = court_points[:, 0]
meter_df['ball_meter_y'] = court_points[:, 1]

# --- Transform player 1 pixel -> meter ---
playerR_points = ballAndPlayer_df[['player_1_x', 'player_1_y']].to_numpy(dtype=np.float32)
playerR_points_reshaped = playerR_points.reshape(-1, 1, 2)
playerR_court_points = cv2.perspectiveTransform(playerR_points_reshaped, M)
playerR_court_points = playerR_court_points.reshape(-1, 2)
meter_df['player_1_x'] = playerR_court_points[:, 0]
meter_df['player_1_y'] = playerR_court_points[:, 1]


# --- Transform player 2 pixel -> meter ---
playerL_points = ballAndPlayer_df[['player_2_x', 'player_2_y']].to_numpy(dtype=np.float32)
playerL_points_reshaped = playerL_points.reshape(-1, 1, 2)
playerL_court_points = cv2.perspectiveTransform(playerL_points_reshaped, M)
playerL_court_points = playerL_court_points.reshape(-1, 2)
meter_df['player_2_x'] = playerL_court_points[:, 0]
meter_df['player_2_y'] = playerL_court_points[:, 1]

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
    """Full singles court markings in meter-space -- these are fixed ITF
    dimensions, so they don't depend on which points got detected in pixel
    space; only the outer rectangle's corners came from the homography."""
    if fill_color:
        ax.fill([0, COURT_WIDTH, COURT_WIDTH, 0], [0, 0, COURT_LENGTH, COURT_LENGTH],
                color=fill_color, zorder=zorder - 1)
    ax.plot([0, COURT_WIDTH, COURT_WIDTH, 0, 0], [0, 0, COURT_LENGTH, COURT_LENGTH, 0],
            color='#E8EDF2', linewidth=1, zorder=zorder)
    ax.plot([0, COURT_WIDTH], [NEAR_SERVICE_Y, NEAR_SERVICE_Y], color='#E8EDF2', linewidth=1, zorder=zorder)
    ax.plot([0, COURT_WIDTH], [FAR_SERVICE_Y, FAR_SERVICE_Y], color='#E8EDF2', linewidth=1, zorder=zorder)
    ax.plot([CENTER_X, CENTER_X], [NEAR_SERVICE_Y, FAR_SERVICE_Y], color='#E8EDF2', linewidth=1, zorder=zorder)
    ax.plot([0, COURT_WIDTH], [NET_Y, NET_Y], color='dimgray', linewidth=2, zorder=zorder)
    ax.plot([CENTER_X, CENTER_X], [0, CENTER_MARK_LENGTH], color='#E8EDF2', linewidth=1, zorder=zorder)
    ax.plot([CENTER_X, CENTER_X], [COURT_LENGTH - CENTER_MARK_LENGTH, COURT_LENGTH],
            color='#E8EDF2', linewidth=1, zorder=zorder)


# Court gets its own axes, and the frame/time label + legend get a separate
# narrow side panel axes -- not just a margin within the court's data space.
# A margin approach (tried previously) still shares the court's coordinate
# range, so (a) it competes for room with player tracking data, which can
# legitimately extend behind the baseline, and (b) its exact size needed to
# avoid the figure edge clipping it depends on the window's actual render
# size, which isn't reliable across interactive backends. A separate axes
# sidesteps both: it has its own space that court data can never enter.
fig = plt.figure(figsize=(6, 8))
grid = fig.add_gridspec(1, 2, width_ratios=[4, 1.3], wspace=0.05)
mapping_window = fig.add_subplot(grid[0, 0])
info_ax = fig.add_subplot(grid[0, 1])
fig.canvas.manager.set_window_title('2D Mapping Window')

# Everything below is static across frames -- draw it once instead of
# rebuilding it (mapping_window.clear() + redraw) on every single frame,
# which was costing ~110ms/frame and made playback ~7x slower than the
# source video. Only the trail positions and time text change per frame.
draw_court(mapping_window, fill_color='#133458')
# Wider than just the court itself -- players commonly stand a couple meters
# behind the baseline (return position, etc.), and a tight margin was
# clipping them out of view at the top/bottom edge of the plot.
mapping_window.set_xlim(-2, COURT_WIDTH + 2)
mapping_window.set_ylim(-3, COURT_LENGTH + 3)
mapping_window.set_aspect('equal')
mapping_window.set_title('2D Mapping Window')

ball_scatter = mapping_window.scatter([], [], color='#838921', s=40)
player1_scatter = mapping_window.scatter([], [], color='#BD4444', marker='o', s=60)
player2_scatter = mapping_window.scatter([], [], color='#BD4444', marker='^', s=60)

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

    # Driven from here (not a separate loop) so it stays frame-synced with
    # the mapping instead of the two windows running on independent timers
    # and drifting apart. Seeking explicitly rather than relying on
    # sequential reads keeps it correct even if the animation ever calls
    # update() out of strict order (e.g. its initial blit setup pass).
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

information_window[0,0].set_xlim(-1, 9.23)
information_window[0,0].set_ylim(-2, 25)
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

information_window[0,1].set_xlim(-1, 9.23)
information_window[0,1].set_ylim(-2, 25)
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