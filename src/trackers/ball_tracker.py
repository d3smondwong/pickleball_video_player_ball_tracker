import logging
import numpy as np
import pickle
import cv2
import pandas as pd
from ultralytics import YOLO

from supervision.detection.core import Detections
from inference_sdk import InferenceHTTPClient
from src.utils.bbox_utils import get_center_of_bbox

# Set up logging
logger = logging.getLogger(__name__)
class BallTracker:
    def __init__(self, api_key: str, model_id: str):
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        self.model_id = model_id

    def interpolate_ball_positions(self, ball_positions: list[dict[int, list[float]]]) -> list[dict[int, list[float]]]:
        """
        Given a list of per-frame detections where each element is either an empty dict (no detection)
        or a dict mapping the ball track id (expected 0) to a bounding box [x1, y1, x2, y2],
        this function linearly interpolates short gaps in the detected coordinates (up to a few frames)
        and returns a list of the same length where short gaps are filled and long gaps remain empty dicts.
        Args:
            ball_positions (list[dict[int, list[float]]]):
                A list of per-frame detections. Each item should be either {}
                (no detection) or {0: [x1, y1, x2, y2]} representing the ball bounding box.
        Returns:
            list[dict[int, list[float]]]:
                A list with the same length as `ball_positions`. Frames with short missing runs
                are filled by linear interpolation of [x1, y1, x2, y2]; frames that remain
                undetected after interpolation (long gaps) are returned as empty dicts.
        """

        # ball_detections is in the following format:
        # ball_detections: [{0: [523.0, 148.0, 538.0, 162.0]},  {},]
        # where 0 is the track ID and the list contains the bounding box coordinates [x1, y1, x2, y2]
        # {} indicates that no ball was detected in that frame

        # Iterates through the list and extracts the x,y values with the key 0 (where ball is detected). Get empty list if not detected
        # Extract raw rows (preserve original list for early return)
        raw_rows = [frame.get(0, []) for frame in ball_positions]

        # Build DataFrame with NaNs for missing frames
        rows = []
        for r in raw_rows:
            if not r or len(r) < 4:
                rows.append([np.nan, np.nan, np.nan, np.nan])
            else:
                rows.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
        df = pd.DataFrame(rows, columns=['x1', 'y1', 'x2', 'y2'])

        # Parameters
        max_gap_frames = 5           # only interpolate gaps <= this 

        # Interpolate short gaps
        df_interp = df.interpolate(limit=max_gap_frames, limit_direction='both')

        # Convert back: if any coordinate is still NaN -> return {} for that frame. Handles long gaps like ball is out of frame
        ball_positions = []
        for _, r in df_interp.iterrows():
            if r.isnull().any():
                ball_positions.append({})
            else:
                ball_positions.append({0: [round(float(r['x1']), 1), round(float(r['y1']), 1), round(float(r['x2']), 1), round(float(r['y2']), 1)]})
        return ball_positions

    def detect_frame(self, frame: np.ndarray) -> dict[int, list[float]]:
        """
        Detects the ball in a single video frame using the configured YOLO model and returns bounding box coordinates for the detected ball.
        Args:
            frame (np.ndarray): Image array to run inference on (height, width, channels). Must be in the color format and dtype expected by the inference client/model.
        Returns:
            dict[int, list[float]]: A mapping from detection index to bounding box coordinates in the format
                [x_min, y_min, x_max, y_max].
                - If a ball is detected, this will typically be {0: [x_min, y_min, x_max, y_max]} (the first detected box).
                - If no detections meet the confidence threshold or inference yields no boxes, an empty dict is returned.
        Notes:
            - Internally this method calls self.client.infer(...) and converts the result with Detections.from_inference(...).
            - A confidence filter of 0.15 is applied to detections; only boxes with confidence > 0.15 are considered.
            - The model is expected to contain a single class ("ball"); the implementation currently returns only the first detected bounding box.
        """
        # Detect the items (Ball) in the frame using the YOLO model. The only class in this model is "ball".
        # There is only 1 ball in the frame, so we can use predict instead of track
        result = self.client.infer(inference_input=frame, model_id=self.model_id)

        detections = Detections.from_inference(result)
        if detections.confidence is not None:
            detections = detections[detections.confidence > 0.15]
        else:
            detections = Detections.empty()

        # Create a dictionary to store the ball and their bounding boxes
        ball_dict = {}

        # If no boxes are detected, return an empty dictionary
        if detections is None or len(detections) == 0:
            logger.warning("No boxes detected.")
            return ball_dict

        # There is only 1 class (ball) in this model. Store the bounding box coordinates
        if len(detections) > 0:

            bbox = detections.xyxy.tolist()[0]

            ball_dict[0] = bbox

        return ball_dict

    def detect_frames(self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str | None = None
    ) -> list[dict[int, list[float]]]:
        """
        Detect ball positions in a sequence of video frames, optionally using or creating a cached "stub" file.
        Args:
            frames (list[np.ndarray]): A list of video frames (numpy arrays) to process. Each frame will be passed to self.detect_frame to obtain detections.
            read_from_stub (bool, optional): If True and stub_path is provided, attempt to load and return previously saved detections from the stub file instead of processing frames. Defaults to False.
            stub_path (str | None, optional): Path to a pickle file used to load or save cached detections. If provided and read_from_stub is True, the function will try to load detections from this file; if provided and read_from_stub is False, the function will save the new detections to this file after processing. If the stub file is not found when loading, the function logs an error and proceeds to process frames.
        Returns:
            list[dict[int, list[float]]]: A list with one dictionary per input frame. Each dictionary maps detection identifiers (int) to a list of numeric detection properties (list[float]) such as coordinates, size, and/or confidence. The exact contents of the list depend on the implementation of self.detect_frame.
        """
        ball_detections = []

        # If read_from_stub is True, load the ball detections from the stub file
        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, 'rb') as file:
                    ball_detections = pickle.load(file)

                return ball_detections

            except FileNotFoundError:
                logger.error(f"Stub file {stub_path} not found. Returning empty detections.")
                ball_detections = []

        # For each frame, detect the ball and append it to the list
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        # If read_from_stub is False, save the ball detections to the stub file
        if stub_path is not None:
            with open(stub_path, 'wb') as file:
                pickle.dump(ball_detections, file)

        return ball_detections

    def _draw_triangle(self, frame: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int]) -> np.ndarray:
        """
        Draws a filled triangle with an outlined border at the top center of the given bounding box on the frame.
        The triangle is positioned such that its base is at the top center of the bounding box, and its tip points upward.
        The triangle is filled with the specified color and outlined in black.
        Args:
            frame (np.ndarray): The image frame on which to draw the triangle.
            bbox (tuple[int, int, int, int]): The bounding box coordinates in the format (x1, y1, x2, y2).
            color (tuple[int, int, int]): The BGR color tuple for filling the triangle.
        Returns:
            np.ndarray: The frame with the triangle drawn on it.
        """
        x1, y1, x2, y2 = map(int, bbox)
        x_center, y_center = get_center_of_bbox(bbox)

        # Define triangle parameters
        triangle_points = np.array([
                [x_center,y1],
                [x_center-5,y1-10],
                [x_center+5,y1-10],
            ], dtype=np.int32)

        # Draw the triangle on the frame
        cv2.fillPoly(frame, [triangle_points], color)
        cv2.polylines(frame, [triangle_points], isClosed=True, color=(0, 0, 255), thickness=1)

        return frame

    def draw_bounding_boxes(self, video_frames: list[np.ndarray], ball_detections: list[dict[int, list[float]]]) -> list[np.ndarray]:
        """
        Draws a circle at the center of each detected bounding box and annotates each detection
        with a triangle (via self._draw_triangle). Processes frames and detection dicts pairwise.
        Args:
            video_frames (list[np.ndarray]): List of image frames (H, W, 3) in BGR color space.
                Frames may be modified in-place by drawing operations.
            ball_detections (list[dict[int, list[float]]]): List of dictionaries, one per frame.
                Each dictionary maps a track_id (int) to a bounding box specified as
                [x1, y1, x2, y2] (float or int pixel coordinates, top-left and bottom-right).
        Returns:
            list[np.ndarray]: The list of frames with drawn annotations. The returned list contains
            the frames that were iterated (iteration stops at the shorter length of the two input lists),
            and the frames themselves are the same objects from video_frames (mutated in-place).
        Notes:
            - A red circle is drawn at the bounding box center; the radius is computed as
              max(2, int(max(width, height) * 0.4)).
            - Coordinates are cast to integers for drawing functions.
            - Triangles are drawn using self._draw_triangle(frame, bbox, (0, 0, 255)).
        """
        output_video_frames = []

        # Iterate through the video frames and ball detections. zip combines the two lists so that we can iterate through both at the same time
        for frame, ball_dict in zip(video_frames, ball_detections):

            # Draw Bounding Boxes. loop over the ball_dict which contains the track ID and bounding box coordinates
            for track_id, bbox in ball_dict.items():

                # Extract coordinates from the bounding box
                x1, y1, x2, y2 = bbox

                # Draw a circle at the centre of the bounding box to represent the ball
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                width = int(abs(x2 - x1))
                height = int(abs(y2 - y1))

                # radius based on bbox size (adjust factor as needed)
                radius = max(2, int(max(width, height) * 0.4))
                cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), 1) # Red Circle

            # Draw triangle to annotate ball
            for track_id, bbox in ball_dict.items():
                frame = self._draw_triangle(frame, bbox,(0, 0, 255))  # Red triangle

            output_video_frames.append(frame)

        return output_video_frames