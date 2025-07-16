from ultralytics import YOLO
import numpy as np
import pickle
import cv2

class PlayerTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect_frame(self, frame: np.ndarray) -> dict[int, list[float]]:
        """
        Detects and tracks players (persons) in a given video frame using a YOLO model.
        Args:
            frame (numpy.ndarray): The input video frame to process.
        Returns:
            dict: A dictionary mapping player track IDs (int) to their bounding box coordinates (list of float).
                  Only includes detected objects classified as "person".
                  If no persons are detected, returns an empty dictionary.
        Notes:
            - Uses the YOLO model's tracking functionality to detect and assign track IDs to objects.
            - Only bounding boxes with valid track IDs, coordinates, and class information are considered.
            - Prints a message if no bounding boxes are detected in the frame.
        """

        # Detect the items (eg. person) in the frame using the YOLO model
        results = self.model.track(frame, persist=True)[0]

        # Extract the bounding boxes and their track IDs
        id_name_dict = results.names

        # Create a dictionary to store the player track IDs and their bounding boxes
        player_dict = {}

        # If no boxes are detected, return an empty dictionary
        if results.boxes is None:
            print(f"No boxes detected in {results.names}.")
            return player_dict

        # Focus on person. Iterate through the detected boxes and store the track ID and bounding box coordinates
        for box in results.boxes:

            # If the box is empty, skip it
            if box.id is None or len(box.id) == 0:
                continue

            # If the box has no bounding box coordinates, skip it
            if box.xyxy is None or len(box.xyxy) == 0:
                continue

            # If the box has no class ID, skip it
            if box.cls is None or len(box.cls) == 0:
                continue

            # If the box has no class name, skip it
            if id_name_dict is None or len(id_name_dict) == 0:
                continue

            # Extract the track ID from the box
            track_id = int(box.id.tolist()[0])

            # Get the bounding box coordinates
            result = box.xyxy.tolist()[0]

            # Get the class ID and name from the box
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]

            # If the detected object is a person, store the track ID and bounding box coordinates
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def detect_frames(

        self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str | None = None
    ) -> list[dict[int, list[float]]]:
        """
        Detects players in a list of video frames and returns their detections.
        This method processes each frame to detect players and returns a list of detection dictionaries,
        where each dictionary maps player IDs to their detected coordinates or features. Optionally,
        detections can be loaded from or saved to a stub file using pickle serialization.
    
        Args:
            frames (list[np.ndarray]): List of video frames (as numpy arrays) to process.
            read_from_stub (bool, optional): If True, loads detections from the stub file instead of processing frames. Defaults to False.
            stub_path (str | None, optional): Path to the stub file for loading or saving detections. If None, stub functionality is disabled.
        Returns:
            list[dict[int, list[float]]]: A list where each element is a dictionary mapping player IDs to lists of detected features for each frame.
        """

        player_detections = []

        # If read_from_stub is True, load the player detections from the stub file
        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, 'rb') as file:
                    player_detections = pickle.load(file)

                return player_detections

            except FileNotFoundError:
                print(f"Stub file {stub_path} not found. Returning empty detections.")
                player_detections = []

        # For each frame, detect the player and append it to the list
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        # If read_from_stub is False, save the player detections to the stub file
        if stub_path is not None:
            with open(stub_path, 'wb') as file:
                pickle.dump(player_detections, file)

        return player_detections

    def draw_bounding_boxes(self, video_frames, player_detections):
        output_video_frames = []

        # Iterate through the video frames and player detections. zip combines the two lists so that we can iterate through both at the same time
        for frame, player_dict in zip(video_frames, player_detections):

            # Draw Bounding Boxes. loop over the player_dict which contains the track ID and bounding box coordinates
            for track_id, bbox in player_dict.items():

                # Extract coordinates from the bounding box
                x1, y1, x2, y2 = bbox

                # Draw the bounding box and player ID on the frame
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames