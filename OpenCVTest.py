import cv2
import numpy as np
import pandas as pd

#画像クロッピング: 写真
'''
img = cv2.imread('test_image.png')  # Load an image from file
cropped_frame = img[0:1000, 100:350]

cv2.rectangle(cropped_frame, (90, 90), (160,160), (0, 255, 0), 5)  # Draw a rectangle on the cropped frame
cv2.putText(cropped_frame, 'Cropped Frame', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Add text to the cropped frame

cv2.imshow("Original Image", img)  # Display the original image
cv2.imshow("Cropped Frame", cropped_frame)  # Display the cropped frame with rectangle
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all OpenCV windowsD:\TennisProject\Practice\OpenCVTest.py
'''

#画像クロッピング: 動画
'''
cap = cv2.VideoCapture('test_video.mp4')  # ファイルから動画を読み込む
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
while cap.isOpened():
    succes, frame = cap.read()  # 動画からフレームを読み込む
    if not succes:
        print("End  of video or cannot read the frame.")
        break
    
    resized_frame = cv2.resize(frame, (720, 1080))  # Scale to 1280x720
    cv2.imshow("Video Playback", resized_frame) # 現在のフレームを表示する  
    if cv2.waitKey(500) & 0xFF == ord('q'):  # 'q'キーが押されたらループを終了する
        break

cap.release()  # 動画ファイルを解放(かいほう)する
cv2.destroyAllWindows()  # すべてのOpenCVウィンドウを閉じる
'''

#DataFrameの作成 example1
'''
tracking_data = np.array([
    [1, 100, 800, 105, 790],
    [2, 110, 790, 150, 600],
    [3, 120, 780, 200, 450],
    [4, 130, 770, 250, 300],
    [5, 140, 760, 300, 150]
])

print("--- Raw NumPy Array ---")
print(tracking_data)
print(f"Array Shape: {tracking_data.shape}\n")


ื#name the column names for the DataFrame
columns = ['Frame', 'Player_X', 'Player_Y', 'Ball_X', 'Ball_Y']
df = pd.DataFrame(data=tracking_data, columns=columns)

df.set_index('Frame', inplace=True)

print("--- DataFrame ---")
print(df)
print("\n")

# --- STEP 3: DATA MANIPULATION ---
# Now that our data is aligned in a DataFrame, we can easily manipulate it!

# 1. Extract just the Player's trajectory
player_trajectory = df[['Player_X', 'Player_Y']]
print("--- Player Trajectory Only ---")
print(player_trajectory)
print("\n")

# 2. Filter data (e.g., Find frames where the Ball's Y coordinate crossed the net at Y=500)
net_crossing = df[df['Ball_Y'] < 500]
print("--- Frames where the ball is past the net (Y < 500) ---")
print(net_crossing)
'''

#DataFrameの作成 example2
raw_data = np.array([
    [1, 10.0, 20.0, 50.0, 80.0],
    [2, 12.0, 22.0, 55.0, 75.0],
    [3, 15.0, 25.0, 65.0, 65.0],
    [4, 19.0, 29.0, 80.0, 50.0],
    [5, 20.0, 30.0, 100.0, 30.0],
    [6, 20.0, 30.0, 130.0, 10.0]
])

df = pd.DataFrame(data=raw_data, columns=['Frame', 'Player_1X', 'Player_1Y', 'Ball_X', 'Ball_Y'])
df.set_index('Frame', inplace=True)
print(df)

#--1.The Distance Metic, Total Distance P1 did all 6 frames--
def distance_compute(df, x_col='Player_1X' , y_col = 'Player_1Y'):
    coord = df[[x_col, y_col]].to_numpy(dtype=float)
    if coord.shape[0] < 2:
        return 0.0
    diff = np.diff(coord, axis=0)
    dist = np.sqrt((diff**2).sum(axis=1)) #sqrt(diff_x_col**2 + diff_y_col**2)
    total = float(dist.sum())
    return total,dist

total, dist = distance_compute(df, x_col='Player_1X', y_col='Player_1Y')
print(f"Total Distance Player 1 traveled over 6 frames: {total:.2f} units")
print("Movement between each consecutive frame:")

for i,d in enumerate(dist, start=1):
    print(f"Frame {i} to Frame {i+1}: {d: .2f} units")


#--2.The Ball Distance--
def ball_distance_compute(df, x_col='Ball_X', y_col='Ball_Y'):
    coord = df[[x_col, y_col]].to_numpy(dtype=float) #To numpy array
    if coord.shape[0] < 2:
        return 0.0
    diff = np.diff(coord, axis=0)
    dist = np.sqrt((diff**2).sum(axis=1)) #sqrt(diff_x_col**2 + diff_y_col**2)
    total = float(dist.sum())
    return total, dist

ball_total, ball_dist = ball_distance_compute(df, x_col='Ball_X', y_col='Ball_Y')
print(f"Total Distance the ball traveled over 6 frames: {ball_total:.2f} units")
print("Ball movement between each consecutive frame:")
for i, d in enumerate(ball_dist, start=1):
    print(f"Frame {i} to Frame {i+1}: {d:.2f} units")

max_index = int(np.argmax(ball_dist)) #max frame can be 1 but array make in be 0 so we add 1 to make it correct
max_frame_pair = f"Frame {max_index+1} to Frame {max_index}"
print(f"The biggest ball movement happened at {max_frame_pair} with {ball_dist[max_index]:.2f} units")

#DataFrameの作成 example2
