import numpy as np
import pickle
import cv2
import logging

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO
from src.utils.bbox_utils import measure_distance, get_center_of_bbox, get_bbox_width, get_foot_position, measure_xy_distance

# Set up logging
logger = logging.getLogger(__name__)
class PlayerTracker:
    def __init__(self, model_path: str, cfg: DictConfig):
        self.model = YOLO(model_path)
        self.tracker_cfg = cfg.player_tracker
        self.player_names = self._parse_player_names_from_cfg(cfg)

    @staticmethod
    def _parse_player_names_from_cfg(cfg: DictConfig) -> dict[int, str]:
        """
        Convert cfg.player_details (DictConfig) into a mapping {player_id: name}.
        Handles either explicit 'player_id' field or keys like 'player_1'.
        """
        if cfg is None or getattr(cfg, "player_details", None) is None:
            return {}

        pd = OmegaConf.to_container(cfg.player_details, resolve=True)
        names: dict[int, str] = {}
        for key, val in pd.items():
            try:
                pid = val.get("player_id")
                if pid is None:
                    m = re.search(r'(\d+)$', str(key))
                    if not m:
                        continue
                    pid = int(m.group(1))
                else:
                    pid = int(pid)
                name = val.get("name")
                if name:
                    names[pid] = str(name)
            except Exception:
                continue
        return names


    def point_in_polygon(self, point: tuple[float, float], polygon_vertices: list[tuple[float, float]]) -> bool:
        """
        Determines if a 2D point is inside a polygon using the Ray Casting Algorithm (or even-odd rule).

        Args:
            point (tuple[float, float]): The (x, y) coordinates of the test point (e.g., player's foot position).
            polygon_vertices (list[tuple[float, float]]): An ordered list of the polygon's (x, y) vertices
                                                        (e.g., the court's perimeter keypoints).

        Returns:
            bool: True if the point is strictly inside the polygon, False otherwise.
        """
        x, y = point
        n = len(polygon_vertices)
        inside = False

        # A polygon must have at least 3 vertices
        if n < 3:
            return False

        # Start with the last vertex as the previous vertex
        p1x, p1y = polygon_vertices[n - 1]

        # Iterate through all edges of the polygon
        for i in range(n):
            p2x, p2y = polygon_vertices[i]

            # Check if the ray (horizontal line from 'point' to infinity) intersects the edge (p1, p2)
            # The core logic checks two conditions:
            # 1. The y-coordinate of the test point must be between the y-coordinates of the two vertices (p1y, p2y).
            # 2. The intersection point of the ray with the edge must be to the right of the test point's x-coordinate.
            if ((p1y > y) != (p2y > y)):
                # Calculate the x-coordinate of the intersection point
                # Equation for x-intersection: x_int = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
                # We check if x is less than x_int (i.e., the intersection is to the right of the test point)
                intersect_x = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x

                if x < intersect_x:
                    # If an intersection is found, flip the 'inside' flag
                    inside = not inside

            # Move to the next edge
            p1x, p1y = p2x, p2y

        return inside

    def expand_buffer_zone_polygon(self, polygon_vertices: list[tuple[float, float]], scale: float = 1.2) -> list[tuple[float, float]]:
        """
        Expands a polygon by scaling its vertices outward from the centroid.
        Args:
            polygon_vertices (list[tuple[float, float]]):
                A list of (x, y) tuples representing the vertices of the polygon.
            scale (float, optional):
                The factor by which to scale the polygon outward from its centroid.
                Default is 1.2.
        Returns:
            list[tuple[float, float]]:
                A list of (x, y) tuples representing the expanded polygon vertices.
        Notes:
            - If polygon_vertices is empty, returns an empty list.
            - The expansion is performed by calculating the centroid of the polygon,
              then scaling each vertex's vector from the centroid by the given scale factor.
        """

        if not polygon_vertices:
            return []

        cx = sum(p[0] for p in polygon_vertices) / len(polygon_vertices)
        cy = sum(p[1] for p in polygon_vertices) / len(polygon_vertices)

        expanded_polygon = []
        for x, y in polygon_vertices:
            vx = x - cx
            vy = y - cy
            # scale vector outward from centroid
            nx = cx + vx * scale
            ny = cy + vy * scale
            expanded_polygon.append((float(nx), float(ny)))
        return expanded_polygon

    def choose_players(
        self,
        court_keypoints: list,
        player_dict: dict[int, list[float]]
    ) -> list[int]:

        """
        Selects up to four player track IDs whose foot positions are closest to the court keypoints and are located within an expanded buffer zone around the court.
        Args:
            court_keypoints (list): List of court keypoints, each represented as either a dictionary with 'x' and 'y' keys or a tuple/list of (x, y) coordinates.
            player_dict (dict[int, list[float]]): Dictionary mapping player track IDs to their bounding box coordinates.
        Returns:
            list[int]: List of up to four selected player track IDs, sorted by proximity to the court keypoints.
        Raises:
            ValueError: If required court keypoints are missing or have invalid formats.
        Notes:
            - The buffer zone polygon is created by expanding the court corners by 20% to ensure players near the court are included.
            - Only players whose foot positions are inside the buffer zone are considered.
            - The selection is based on the minimum distance from each player's foot position to any court keypoint.
        """

        # Normalize court corner keypoints into (x, y) float tuples so point_in_polygon can unpack them
        court_corners = []

        # Ray casting algorithm requires the polygon vertices to be ordered. Using counter-clockwise [0, 1, 2, 3]
        for i in [0, 1, 2, 3]:
            try:
                kp = court_keypoints[i]
            except Exception:
                raise ValueError(f"court_keypoints missing index {i}: {court_keypoints}")

            if isinstance(kp, dict):
                try:
                    x = float(kp.get("x"))
                    y = float(kp.get("y"))
                except Exception:
                    raise ValueError(f"Invalid keypoint at index {i}: {kp}")
                court_corners.append((x, y))
            else:
                try:
                    x, y = kp[0], kp[1]
                    court_corners.append((float(x), float(y)))
                except Exception:
                    raise ValueError(f"Invalid keypoint format at index {i}: {kp}")

        # create a buffered polygon (20% larger) and use it for inside-court checks
        buffer_zone_corners = self.expand_buffer_zone_polygon(court_corners, scale=1.2)

        distances = []
        for track_id, bbox in player_dict.items():

            # get the player foot position
            player_foot_position = get_foot_position(bbox)

            # Check if the player's foot is inside the court + buffer zone polygon
            if not self.point_in_polygon(player_foot_position, buffer_zone_corners):
                continue

            # Calculate the distance from the player foot position to each court keypoint
            min_distance = float('inf')

            # Loop through the court keypoints and calculate the distance. Court keypoints are in the format [x1, y1, x2, y2, ...] so we skip every 2nd element
            for kp in court_keypoints:
                if isinstance(kp, dict):
                    court_keypoint = (kp['x'], kp['y'])
                else:
                    court_keypoint = kp  # fallback if already a tuple/list
                distance = measure_distance(player_foot_position, court_keypoint)
                if distance < min_distance:
                    min_distance = distance

            distances.append((track_id, min_distance))

        # sort the distances in ascending order
        distances.sort(key = lambda x: x[1])

        # Choose the first 4 players with the smallest distances
        chosen_players = [distances[i][0] for i in range(min(4, len(distances)))]
        return chosen_players

    def choose_and_filter_players(
        self,
        court_keypoints: list,
        player_detections: list[dict[int, list[float]]]
    ) -> list[dict[int, list[float]]]:
        """
        Filters player detections for each frame by selecting the closest players to the court keypoints.
        Args:
            court_keypoints (list or np.ndarray): The keypoints representing the court for each frame.
            player_detections (list of dict): A list where each element is a dictionary mapping player track IDs to bounding boxes for a frame.
        Returns:
            list of dict: A list of dictionaries, each containing only the selected player detections for each frame, filtered to include only the closest players as determined by `choose_players`.
        """
        filtered_player_detections = []

        for player_dict in player_detections:
            if player_dict:
                # Dynamically choose the closest players for this frame
                chosen_players = self.choose_players(court_keypoints, player_dict)
                filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            else:
                filtered_player_dict = {}
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections

    def detect_frame(self, frame: np.ndarray) -> dict[int, list[float]]:
        """
        Detects and tracks players (persons) in a given video frame using a YOLO model. Pass in bytetrack config for better tracking across frames.
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
        results = self.model.track(frame, persist=True, tracker=self.tracker_cfg)[0]

        # Extract the bounding boxes and their track IDs
        id_name_dict = results.names

        # Create a dictionary to store the player track IDs and their bounding boxes
        player_dict = {}

        # If no boxes are detected, return an empty dictionary
        if results.boxes is None:
            logger.error(f"No boxes detected in {results.names}.")
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
                logger.error(f"Stub file {stub_path} not found. Returning empty detections.")
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

    def _draw_ellipse(
        self,
        frame: 'np.ndarray',
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
        track_id: int | None
    ) -> 'np.ndarray':

        x1, y1, x2, y2 = map(int, bbox)
        x_center, y_center = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        ###
        # Drawing the ellipse to represent the player or referee
        ###

        # Define ellipse parameters
        axes_length = (int(width), int(0.35*width))
        angle = 0
        # Ellipse is drawn from start_angle to end_angle. Draw -45 - 235 degrees for players to stand out
        start_angle = -45
        end_angle = 235
        thickness = 2
        line_type = cv2.LINE_4

        # Draw the ellipse on the frame
        cv2.ellipse(frame, (int(x_center), int(y2)), axes_length, angle, start_angle, end_angle, color, thickness, line_type)

        ###
        # Drawing the rectangle place the player number
        ###

        # Define rectangle parameters
        rectangle_width = max(60, int(width * 0.8))
        rectangle_height = max(20, int(width * 0.30))

        # Position rectangle slightly below the player's feet
        rectangle_x1 = int(x_center - rectangle_width // 2)
        rectangle_y1 = int(y2 + 6)
        rectangle_x2 = int(x_center + rectangle_width // 2)
        rectangle_y2 = int(rectangle_y1 + rectangle_height)

        # Text parameters for player id/name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(0.8, width / 200.0))
        color_text = (0, 0, 0)
        thickness_text = 2

        # Draw rectangle and player id/name if provided
        if track_id is not None:
            # Draw filled rectangle as background for text
            cv2.rectangle(frame, (rectangle_x1, rectangle_y1), (rectangle_x2, rectangle_y2), color, cv2.FILLED)

            # Determine display text: prefer configured player name, fallback to numeric id
            display_text = self.player_names.get(track_id, str(track_id))

            # Center text horizontally in the rectangle and clamp inside boundaries
            (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness_text)
            x_text = int(x_center - text_w // 2)
            x_text = max(rectangle_x1 + 2, min(x_text, rectangle_x2 - text_w - 2))
            y_text = int(rectangle_y1 + (rectangle_height + text_h) // 2)

            # Draw the text
            cv2.putText(frame, display_text, (x_text, y_text), font, font_scale, color_text, thickness_text)

        return frame

    def draw_bounding_boxes(
        self,
        video_frames: list[np.ndarray],
        player_detections: list[dict[int, list[float]]]
    ) -> list[np.ndarray]:
        """
        Draws bounding boxes and player IDs on each video frame based on player detections.
        Args:
            video_frames (list of np.ndarray): List of video frames (images) to annotate.
            player_detections (list of dict): List of dictionaries for each frame, where each dictionary maps
                player track IDs (int) to bounding box coordinates (tuple of four ints: (x1, y1, x2, y2)).
        Returns:
            list of np.ndarray: List of video frames with bounding boxes and player IDs drawn.
        """
        output_video_frames = []

        # Iterate through the video frames and player detections. zip combines the two lists so that we can iterate through both at the same time
        for frame, player_dict in zip(video_frames, player_detections):

            # Draw Bounding Boxes. loop over the player_dict which contains the track ID and bounding box coordinates
            for track_id, bbox in player_dict.items():

                # Extract coordinates from the bounding box
                x1, y1, x2, y2 = bbox

                # Draw an ellipse to represent the player
                frame = self._draw_ellipse(frame, (int(x1), int(y1), int(x2), int(y2)), (0, 255, 0), track_id)

                # Draw the bounding box and player ID on the frame. Using Red Bounding Box
                # cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames