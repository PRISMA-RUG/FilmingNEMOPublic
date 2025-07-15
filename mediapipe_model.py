import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# This dictionary has the keypoints for the mediapipe layout
MEDIAPIPE_ESTIMATOR_KEYPOINTS =  {
    0: "nose",
    1: "left eye (inner)",
    2: "left eye",
    3: "left eye (outer)",
    4: "right eye (inner)",
    5: "right eye",
    6: "right eye (outer)",
    7: "left ear",
    8: "right ear",
    9: "mouth (left)",
    10: "mouth (right)",
    11: "left shoulder",
    12: "right shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right wrist",
    17: "left pinky",
    18: "right pinky",
    19: "left index",
    20: "right index",
    21: "left thumb",
    22: "right thumb",
    23: "left hip",
    24: "right hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
    29: "left heel",
    30: "right heel",
    31: "left foot index",
    32: "right foot index"
}

# Function: load_model_mp
# Params: folder (default "MediaPipe/"). Folder (relative) where the program will look for the mediapipe models
#         type (default "full"). One of  "full" or "lite". Model weight to be loaded.
# Output: MediaPipe Model
def load_model_mp(folder = "MediaPipe/", type = "full"):
    path = folder + "pose_landmarker_" + type +".task"
    b_options = python.BaseOptions(
        model_asset_path=path,
    )
    options = vision.PoseLandmarkerOptions(
        base_options=b_options,
        running_mode=vision.RunningMode.VIDEO,

    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    return landmarker



