import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from scipy.signal import find_peaks
import cv2

court = pd.read_csv(r'D:\TennisProject\court_coordinates_2.csv')

ballAndPlayer_df = pd.read_csv(r'D:\TennisProject\player_tracking_output_1.csv') 

print(ballAndPlayer_df['frame'].dtype)


fps = 60
frame_gap = 1 
frame_time = 1/30
all_data = pd.merge(ballAndPlayer_df, court, how='left') #Because the court data is static so we cross it with the other data
all_data = all_data.ffill().bfill()
print(ballAndPlayer_df)
print(all_data)

#So that we know the distance of the player or the ball in each frame

#To use cv2 Persoective Transform, the attribute is here
src_points = np.array([
    [all_data['BL_x'].mean(), all_data['BL_y'].mean()],
    [all_data['BR_x'].mean(), all_data['BR_y'].mean()],
    [all_data['TR_x'].mean(), all_data['TR_y'].mean()],
    [all_data['TL_x'].mean(), all_data['TL_y'].mean()],
], dtype=np.float32)

#These 4 points is to map to real-world court, the standard tennis singles court is 8.23mx23.77m
dst_points = np.array([
    [0,0],
    [8.23,0],
    [8.23,23.77],
    [0,23.77],
], dtype=np.float32)

#This one is to map the pixel that we got from the csv file to the real world size of the court
#So the result will be a very small number
M = cv2.getPerspectiveTransform(src_points, dst_points)
print(M)

#Tranform ball pixel to ball mater
points = ball_df[['ball_x','ball_y']].to_numpy(dtype=np.float32)

#Reshape to 3D dimentional for the OpenCV
points_reshaped = points.reshape(-1,1,2)

#Apply the Transfrom
court_points = cv2.perspectiveTransform(points_reshaped, M)

#Reshape back to (N,2) and store in the DataFrame ??? Why do we need to reshape back ----> Reshape back so we can show it in form of DataFrame
court_points = court_points.reshape(-1,2)
meter_df['ball_meter_x'] = court_points[:,0]
meter_df['ball_meter_y'] = court_points[:,1]

#Tranform playerR pixel to playerR mater
playerR_points = ballAndPlayer_df[['player_1_x','player_1_y']].to_numpy(dtype=np.float32)
playerR_points_reshaped = playerR_points.reshape(-1,1,2)
playerR_court_points = cv2.perspectiveTransform(playerR_points_reshaped, M)
playerR_court_points = playerR_court_points.reshape(-1,2)
meter_df['player_1_x'] = playerR_court_points[:,0]
meter_df['player_1_y'] = playerR_court_points[:,1]


#Tranform playerL pixel to playerL mater
playerL_points = ballAndPlayer_df[['player_2_x','player_2_y']].to_numpy(dtype=np.float32)
playerL_points_reshaped = playerL_points.reshape(-1,1,2)
playerL_court_points = cv2.perspectiveTransform(playerL_points_reshaped, M)
playerL_court_points = playerL_court_points.reshape(-1,2)
meter_df['player_2_x'] = playerL_court_points[:,0]
meter_df['player_2_y'] = playerL_court_points[:,1]


#From now on will be meter
#Compute ball Speed
dx_court = np.diff(meter_df['ball_meter_x'])
dy_court = np.diff(meter_df['ball_meter_y'])
distance_meter = np.sqrt(dx_court**2 + dy_court**2)
speed_meter_per_sec = distance_meter / frame_time

speed_meter_per_sec = np.insert(speed_meter_per_sec,0, np.nan)
ball_df['speed_meter_per_sec'] = speed_meter_per_sec

print(ball_df['speed_meter_per_sec'].head())
avg_ball_speed = ball_df['speed_meter_per_sec'].mean()
max_ball_speed = ball_df['speed_meter_per_sec'].max()


#Compute each playerR distance from the ball
meter_df['player_1_to_ball_distance'] = np.sqrt(
    (meter_df['player_1_x'] - meter_df['ball_meter_x'])**2 + 
    (meter_df['player_1_y'] - meter_df['ball_meter_y'])**2
)

#Compute each playerL distance from the ball
meter_df['player_2_to_ball_distance'] = np.sqrt(
    (meter_df['player_2_x'] - meter_df['ball_meter_x'])**2 + 
    (meter_df['player_2_y'] - meter_df['ball_meter_y'])**2
)

print("Meter DataFrame")
#Show the Transfromed data in form of dataframe
print(meter_df[['ball_meter_x','ball_meter_y','player_1_x','player_1_y','player_2_x',
                'player_2_y','player_1_to_ball_distance','player_2_to_ball_distance']].head())


#Picking up frame to map.
#Not doing the whole video yet ----> DOING ---> Done
#frame_row = meter_df.iloc[686]

#Giving the court scale

#The video 2DMapping pipline from  the top view of the court, the ball and the player

total_frames = len(meter_df)
trail_length = 0

court_x = [1.1198831e-15,8.2299995e+00, 8.2299986e+00 ,-0.0000000e+00,1.1198831e-15]
court_y = [0,0,2.3770000e+01,2.3770000e+01,0]

fig, mapping_window = plt.subplots(figsize=(6, 8))
fig.canvas.manager.set_window_title('2D Mapping Window')

def update(frame_idx):
    mapping_window.clear()
    mapping_window.plot(court_x, court_y, color='black', linewidth=1)
    start = max(0, frame_idx - trail_length)
    trail = meter_df.iloc[start:frame_idx + 1]
    n = len(trail)
    for i in range(n):
        alpha = (i + 1) / n
        mapping_window.scatter(trail['ball_meter_x'].iloc[i], trail['ball_meter_y'].iloc[i], color='red', alpha=alpha, s=40)
        mapping_window.scatter(trail['player_1_x'].iloc[i], trail['player_1_y'].iloc[i], color='blue', alpha=alpha, s=60)
        mapping_window.scatter(trail['player_2_x'].iloc[i], trail['player_2_y'].iloc[i], color='green', alpha=alpha, s=60)
    mapping_window.set_xlim(-1, 9.23)
    mapping_window.set_ylim(-2, 25)
    mapping_window.set_aspect('equal')
    current_time = frame_idx / fps
    mapping_window.set_title(f'Frame {frame_idx} / {total_frames}  |  Time: {current_time:.2f}s')
    mapping_window.scatter([], [], color='red', label='Ball')
    mapping_window.scatter([], [], color='blue', label='Player 1')
    mapping_window.scatter([], [], color='green', label='Player 2')
    mapping_window.legend(loc='upper right')


#Plot seaborn for the ball speed

fig2, information_window = plt.subplots(2,2,figsize=(6, 8.5))
fig2.canvas.manager.set_window_title('Information Window')

#วาดกรอบ
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


#information_window[0,1].plot(court_x, court_y, color='black', linewidth=1, zorder=1)
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
    plt.axvline(x=idx, color='red', linestyle='--', alpha=0.6)

plt.xlabel('Frame')
plt.ylabel('Ball speed (m/s)')
plt.title('Ball speed over time, with candidate hit frames marked')

ani = animation.FuncAnimation(fig, update, frames=range(0, total_frames), interval=1000/fps)

#plt.tight_layout()
plt.show()