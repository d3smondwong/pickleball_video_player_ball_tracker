from ultralytics import YOLO
import numpy as np
import pickle
import cv2
from supervision.detection.core import Detections
from inference_sdk import InferenceHTTPClient
import pandas as pd


class BallTracker:
    def __init__(self, api_key: str, model_id: str):
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        self.model_id = model_id

    def interpolate_ball_positions(self, ball_positions):

        # ball_detections is in the following format:
        # ball_detections: [{0: [523.0, 148.0, 538.0, 162.0]},  {},]
        # where 0 is the track ID and the list contains the bounding box coordinates [x1, y1, x2, y2]
        # {} indicates that no ball was detected in that frame

        # Iterates through the list and extracts the x,y values with the key 0 (where ball is detected). Get empty list if not detected
        ball_positions = [x.get(0,[]) for x in ball_positions]

        # convert the list into pandas dataframe. To use interpolate and bfill functions
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # Convert the dataframe back to a list of dictionaries with the track ID as the key
        ball_positions = [{0:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frame(self, frame: np.ndarray) -> dict[int, list[float]]:

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
            print("No boxes detected.")
            return ball_dict

        # There is only 1 class (ball) in this model. Store the bounding box coordinates
        if len(detections) > 0:

            bbox = detections.xyxy.tolist()[0]

            ball_dict[0] = bbox

        return ball_dict

    def detect_frames(
        self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str | None = None
    ) -> list[dict[int, list[float]]]:

        ball_detections = []

        # If read_from_stub is True, load the ball detections from the stub file
        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, 'rb') as file:
                    ball_detections = pickle.load(file)

                return ball_detections

            except FileNotFoundError:
                print(f"Stub file {stub_path} not found. Returning empty detections.")
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

    def draw_bounding_boxes(self, video_frames, ball_detections):
        output_video_frames = []

        # Iterate through the video frames and ball detections. zip combines the two lists so that we can iterate through both at the same time
        for frame, ball_dict in zip(video_frames, ball_detections):

            # Draw Bounding Boxes. loop over the ball_dict which contains the track ID and bounding box coordinates
            for track_id, bbox in ball_dict.items():

                # Extract coordinates from the bounding box
                x1, y1, x2, y2 = bbox

                # Draw the bounding box and ball ID on the frame
                # cv2.putText(frame, f"Ball",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames