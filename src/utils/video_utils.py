import cv2
import numpy as np
from typing import List

def read_video(video_path: str) -> List[np.ndarray]:
    """
    Reads a video file and returns its frames as a list of images.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: A list of frames (numpy.ndarray) extracted from the video.

    Raises:
        FileNotFoundError: If the video file cannot be opened.
        ValueError: If the video contains no frames.
    """
    # Load the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    # Check if the video has frames
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
        raise ValueError(f"The video file {video_path} contains no frames.")

    # Read frames from the video
    frames = []

    # Loop through the video frames and append them to the list
    while True:

        # Read the return value (True or False) and frame from the video
        ret, frame = cap.read()

        # If ret is False, break the loop
        if not ret:
            break
        frames.append(frame)

    # Release the video capture object
    cap.release()

    return frames

def save_video(output_video_frames: List[np.ndarray], output_video_path: str) -> None:
    """
    Saves a list of video frames to a video file.

    Args:
        output_video_frames (list of np.ndarray): List of frames (images) to be written to the video file.
            Each frame should be a NumPy array in BGR format (as used by OpenCV).
        output_video_path (str): Path where the output video file will be saved.

    Notes:
        - The output video will be encoded using the MJPG codec at 24 frames per second.
        - The resolution of the output video is determined by the shape of the first frame in the list.
        - All frames in output_video_frames must have the same dimensions and type.
    """
    # Check if the list of frames is empty
    if not output_video_frames:
        raise ValueError("The list of output video frames is empty.")

    # Check if all frames have the same shape
    first_frame_shape = output_video_frames[0].shape
    for frame in output_video_frames:
        if frame.shape != first_frame_shape:
            raise ValueError("All frames must have the same dimensions and type.")

    # Create a VideoWriter object to write the video file using the MJPG codec
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Use the shape of the first frame to determine the video resolution
    if len(first_frame_shape) < 3 or first_frame_shape[2] != 3:
        raise ValueError("Frames must be in BGR format (3 channels).")

    # Create the VideoWriter object with the specified output path, codec, fps, and frame size
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    # Write each frame to the video file
    if not out.isOpened():
        raise IOError(f"Could not open video file for writing: {output_video_path}")

    # Loop through the frames and write them to the video file
    for frame in output_video_frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()