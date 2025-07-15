import plotly.graph_objects as go
import mediapipe as mp
import numpy as np
import cv2
import json
import scipy
from statsmodels.tsa.tsatools import detrend

def get_landmarks(frames, landmarker, timestamps):
    """ get_landmarks
    Obtains landmark coordinates from a mediapipe landmarker

    Params:
        frames - video frames to analyse
        landmarker - mediapipe model for inference
        timestamps - time (in seconds) where each frame occurs
    Outputs:
        landmarks - a list of landmarks using mediapipe's formatting
    """
    landmarks = []
    result_prev = None
    for i, frame in enumerate(frames):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))
        time = mp.Timestamp.from_seconds(timestamps[i]).value
        result = landmarker.detect_for_video(image, time)
        if not result.pose_landmarks:
            result = result_prev
        landmarks = landmarks + [result]
        result_prev = result

    return landmarks

def paint_landmarks(frames, landmarks):
    """ paint_landmarks
    a function to draw landmarks in each video frame

    Params:
        frames - video frames to analyse
        landmarks - list of landmarks using mediapipe model
    Output:
        frames - video frames with landmarks painted
    """
    vid_size = np.shape(frames[0])
    for i, frame in enumerate(frames):
        landmark_x = np.array([landmark.x for landmark in landmarks[i].pose_landmarks[0][:]])
        landmark_y = np.array([landmark.y for landmark in landmarks[i].pose_landmarks[0][:]])
        x, y = (landmark_x * vid_size[1]).astype(int), (landmark_y * vid_size[0]).astype(int)
        for j in range(0, len(x)):
            cv2.circle(frame, (x[j], y[j]), 8, (255, 0, 0), -1)

    return frames



def plot_landmarks_over_time(axis_data, axis_label, landmark_dict, mode=1):
    """
    Plot the evolution of keypoint coordinates over time for a specific axis.

    Parameters:
        axis_data (list of lists): landmarks
        axis_label (str): Label for the axis ('x', 'y', 'z').
        landmark_dict (dict): Dictionary mapping keypoint indices to their names.
    """
    fig = go.Figure()
    if mode:
        if axis_label == 'x':
            axis_data = [result.pose_landmarks for result in axis_data]
            axis_data = [result[0] for result in axis_data]
            axis_data = [[landmark.x for landmark in result] for result in axis_data]
        elif axis_label == 'y':
            axis_data = [result.pose_landmarks for result in axis_data]
            axis_data = [result[0] for result in axis_data]
            axis_data = [[landmark.y for landmark in result] for result in axis_data]
        elif axis_label == 'z':
            axis_data = [result.pose_landmarks for result in axis_data]
            axis_data = [result[0] for result in axis_data]
            axis_data = [[landmark.z for landmark in result] for result in axis_data]
        else:
            return None

    axis_data = list(zip(*axis_data))
    # Add a line for each keypoint
    for idx, keypoint_data in enumerate(axis_data):
        fig.add_trace(go.Scatter(
            x=list(range(len(keypoint_data))),  # Frame numbers (x-axis)
            y=keypoint_data,  # Coordinate values over time (y-axis)
            mode='lines',
            name=landmark_dict.get(idx, f"Keypoint {idx}")  # Label the line
        ))

    # Customize the figure layout
    fig.update_layout(
        title=f"{axis_label.upper()} Coordinate Evolution Over Time",
        xaxis_title="Frame",
        yaxis_title=f"{axis_label.upper()} Coordinate Value",
        legend_title="Landmarks",
    )

    return fig

def landmarks_save(landmarks, patient, task, model, start, end,
                   width, height, crop_x, crop_y, fps, directory = "landmarks", diagnosis=None):
    """ landmarks_save
    This function saves landmark data and metadata to a json for later analysis.

    Params:
    :param fps: - Speed of capture
    :param landmarks: - list of landmarks using mediapipe model
    :param patient: - Patient ID
    :param task: - Task ID
    :param model: - Model used to capture landmarks
    :param start: - Time start
    :param end:  - Time end
    :param width: - Cropping information (final width)
    :param height:  - Cropping information (final height)
    :param crop_x:  - Cropping information (start width)
    :param crop_y: - Cropping information (start height)

    Output:
    Saves a file with the name patientID_taskID_model.json. This json is a dictionary
    that contains that information plus placeholders for:
    - Diagnosis
    - Severity
    - Metrics

    Landmark information is in the 'x, y, z' keys.
    """
    axis_data = [result.pose_landmarks for result in landmarks]
    axis_data = [result[0] for result in axis_data]
    x = [[landmark.x for landmark in result] for result in axis_data]

    axis_data = [result.pose_landmarks for result in landmarks]
    axis_data = [result[0] for result in axis_data]
    y = [[landmark.y for landmark in result] for result in axis_data]

    axis_data = [result.pose_landmarks for result in landmarks]
    axis_data = [result[0] for result in axis_data]
    z = [[landmark.z for landmark in result] for result in axis_data]

    landmark_data = {
        "patient_id": patient,
        "task_id": task,
        "diagnosis": diagnosis,
        "severity": None,
        "scores": None,
        "model": model,
        "time_start": start,
        "time_end": end,
        "frame_rate": fps,
        "x_crop_data": [crop_x, width],
        "y_crop_data": [crop_y, height],
        "x": x,
        "y": y,
        "z": z,
    }

    with open(f'{directory}/{patient}_{task}_{model}.json', 'w') as f:
        json.dump(landmark_data, f)

    return landmark_data


def peaks_and_consistency(landmark_dict, target_landmark, landmark_names, axis="x", detrend_order=2, peak_distance=2):
    """ peaks_and_consistency
    This function determines an oscillating movement's peaks, then calculates the average difference in duration between
    them to compute the metric "CCY"

    :param landmark_dict: - A dictionary generated by landmarks_save
    :param target_landmark: - Index of landmark to check, for example from mediapipe_model.MEDIAPIPE_ESTIMATOR_KEYPOINTS
    :param landmark_names: - A dictionary of landmark names for display, recommended to load the one above.
    :param axis: - One of 'x', 'y', or 'z'
    :param detrend_order: - Order of polynomial to detrend time series
    :param peak_distance: - Minimum distance between peaks (in frames)

    :return: [ccy, fig], where ccy is the metric and fig is a plot of peaks and the target landmark
    """
    target_series = np.transpose(landmark_dict[axis])[target_landmark]
    target_series = detrend(target_series, detrend_order)
    peaks, _ = scipy.signal.find_peaks(target_series, distance=peak_distance, height=np.mean(target_series))

    dts = []
    for i in range(len(peaks) - 1):
        dts.append(peaks[i + 1] - peaks[i])

    dts = np.array(dts) / landmark_dict["frame_rate"]
    distances = dts - np.mean(dts)
    distances = np.sqrt(distances ** 2)
    ccy = 1 / np.sum(distances)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=peaks, y=target_series[peaks], mode="markers",
                             name="Peaks", marker=go.scatter.Marker(size=10, symbol="diamond")))
    fig.add_trace(go.Scatter(x=[x for x in range(0, len(target_series))], y=target_series, mode='lines',
                             name=landmark_names[target_landmark]))
    fig.update_layout(
        title = {
            "text": f"{landmark_names[target_landmark]} movement through time in {axis} axis",
        },
        xaxis = {
            "title": {"text":"Time (frames)"},
        },
        yaxis = {
            "title": {"text":f"{axis} movement (pixels)"},
        },
    )


    return ccy, fig