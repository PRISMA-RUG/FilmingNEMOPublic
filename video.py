import imageio.v3 as iio
import av
import numpy as np
from itertools import islice


def convert_time_to_frames(fps, start, end):
    """ convert_time_to_frames
    Transforms a time label to a frame label for video cropping

    Params = fps (frames per second, int)
             start (start time, string, in mm:ss format)
             end (end time, string, in mm:ss format)
    Returns = start_frames (frame to start capturing, exclusive)
              end_frames (frame to stop capturing, exclusive)
    """
    start_mins, start_secs = start.split(':')
    end_mins, end_secs = end.split(':')
    start_secs = int(start_mins)*60 + int(start_secs)
    end_secs = int(end_mins)*60 + int(end_secs)

    start_frames = start_secs * fps
    end_frames = end_secs * fps

    return start_frames, end_frames

def pan_and_crop(frame, width=1920, height=1080, offset_x=0, offset_y=0):
    """ pan_and_crop
    Pans and crops a video frame using numpy. It also indirectly converts the array to a numpy
    array.

    Params:
        frame - x * y * c image compatible with numpy
        width - size in pixels for the width
        height - size in pixels for the height
        offset_x - pixel in the width where the image should begin
        offset_y - pixel in the height where the image should begin

    Returns:
        frame - cropped frame (x+offset:width, y+offset:height, c)
    """
    frame = np.array(frame)
    frame = frame[offset_y:height, offset_x:width]
    return frame

def get_single_frame(file, start, width, height, crop_x, crop_y):
    """ load_video
    Uses ImageIO to load a video file and extract a single frame for previewing purposes.

    Params:
        file  - file name to load
        start  - time label to start capturing
        width/height - size in pixels for the width and height
        crop_x/crop_y - size in pixels for width and height start
    Returns:
        frame (list of arrays) - captured video frame
    """
    fps = int(iio.immeta(file, plugin="pyav")["fps"])
    start_mins, start_secs = start.split(':')
    start_secs = int(start_mins) * 60 + int(start_secs)
    start = start_secs*fps
    frame = next(islice(iio.imiter(file), start, None))
    frame = pan_and_crop(frame, width=width, height=height, offset_x=crop_x, offset_y=crop_y)
    return  frame


def load_video(file = "specialist.mp4", start = "00:20", end = "00:30", width=1920, height=1080, crop_x=0, crop_y=0):
    """ load_video
    Uses ImageIO to load a video file and extract frames based on the cropped time frames.

    Params:
        file (default: test file) - file name to load
        start (default: 20s) - time label to start capturing
        end (default: 30s) - time label to stop capturing
    Returns:
        frames (list of arrays) - captured video frames
        fps (int) - frames per second
    """
    print("Starting video frame capture.")
    fps = int(iio.immeta(file, plugin="pyav")["fps"])
    frames = []
    start, end = convert_time_to_frames(fps, start, end)
    for i,frame in enumerate(iio.imiter(file)):
        if i < start:
            pass
        elif i < end:
            frame = pan_and_crop(frame, width=width, height=height, offset_x=crop_x, offset_y=crop_y)
            frames = frames + [frame]
        else:
            break
    timestamps = np.array(range(1,len(frames)+1))
    timestamps = timestamps/60

    print(f"Captured {len(frames)} frames.")

    return  frames, fps, timestamps


def save_video(frames, output_path, fps=30, codec="h264", pixel_format="yuv420p"):
    """
    Save a list of frames as a video using PyAV.

    Args:
        frames (list): A list of frames (numpy arrays in HxWxC format, dtype=uint8).
        output_path (str): Path to the output video file.
        fps (int): Frames per second of the output video.
        codec (str): Video codec to use (default is "libx264").
        pixel_format (str): Pixel format for the output video (default is "yuv420p").
    """
    # Create an output container
    print("SAVING "+output_path+"...")
    container = av.open(output_path, mode="w")

    # Determine video dimensions from the first frame
    height, width, channels = frames[0].shape
    assert channels == 3, "Frames must be in RGB format (HxWx3)."

    # Add a video stream to the container
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pixel_format

    # Write frames to the video stream
    for frame in frames:
        # Convert the numpy frame to an AVFrame
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        packet = stream.encode(av_frame)
        if packet:
            container.mux(packet)

    # Finalize encoding
    packet = stream.encode(None)
    if packet:
        container.mux(packet)

    # Close the container
    print("Done!")
    container.close()




