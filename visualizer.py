import cv2
import json
import numpy as np
from video import save_video

# Configuration
ldmrks = json.load(open("landmarks/024_13_full.json"))
output_path = "white_frames_video.mp4"
frame_width = 1920
frame_height = 1080
frame_rate = ldmrks["frame_rate"]
dur = 8
num_frames = dur*frame_rate  # 10 seconds at 30 FPS

# Create a white frame
white_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

frames = []
for i in range(num_frames):
    # Create a white frame
    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

    # Extract landmark coordinates for the current frame
    landmark_x = np.array(ldmrks["x"][i])
    landmark_y = np.array(ldmrks["y"][i])

    # Scale coordinates to fit the frame size
    x, y = (landmark_x * frame_height).astype(int), (landmark_y * frame_width).astype(int)

    # Draw each landmark as a blue dot
    for j in range(len(x)):
        cv2.circle(frame, (x[j], y[j]), 8, (255, 0, 0), -1)  # Blue dots

    # Append the frame to the list
    frames.append(frame)

save_video(frames, "visualize.mp4",fps=frame_rate)
