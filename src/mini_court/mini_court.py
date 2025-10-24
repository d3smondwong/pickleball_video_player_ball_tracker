import cv2
import logging
import numpy as np
from pathlib import Path
import pickle
from typing import Tuple, List, Dict, Union

from src.utils.conversions import (
    convert_feet_to_pixel_distance,
    convert_pixel_distance_to_feet,
)
from src.utils.bbox_utils import (
    get_center_of_bbox,
    measure_distance,
    get_foot_position,
    measure_xy_distance,
    get_height_of_bbox,
    get_closest_keypoint_index,
)

# Set up a local logger for MiniCourt actions
logger = logging.getLogger(__name__)


class MiniCourt:
    def __init__(self, frame: np.ndarray):

        # Unit Conversion Constants
        self.FEET_TO_METERS = 0.3048  # 1 foot = 0.3048 meters
        self.KPH_FACTOR = 3.6  # 1 m/s = 3.6 km/h

        # Pickleball court dimensions: 20 ft wide x 44 ft long
        self.PICKLEBALL_COURT_WIDTH_feet = 20
        self.PICKLEBALL_COURT_LENGTH_feet = 44
        # No Volley Zone (Kitchen) dimensions: 7 ft from each side of the net
        self.PICKLEBALL_KITCHEN_LENGTH_feet = 7
        # Pickleball net height: 36 inches (3 ft)
        self.PICKLEBALL_NET_HEIGHT_feet = 3

        self.PLAYER_HEIGHTS_FEET = {
            1: 5.74,  # Example: Parenteau Player 1 is 5 feet 5 inches
            2: 5.92,  # Example: McGuffin Player 2 is 5 feet 10 inches
            9: 5.58,  # Example: Todd Player 9 is 5 feet 7 inches
            15: 6.07,  # Example: Newman Player 15 is 6 feet 2 inches
        }

        # Real-world court coordinates (X_feet, Y_feet) where (0, 0) is the Top-Left corner (Far Baseline).
        self.keypoint_feet_mapping = {
            0: (0.0, 0.0),  # TL Baseline
            1: (20.0, 0.0),  # TR Baseline
            2: (20.0, 44.0),  # BR Baseline
            3: (0.0, 44.0),  # BL Baseline
            4: (0.0, 22.0),  # Net Left
            5: (20.0, 22.0),  # Net Right
            6: (0.0, 15.0),  # Far Kitchen Left
            7: (20.0, 15.0),  # Far Kitchen Right
            8: (0.0, 29.0),  # Near Kitchen Left
            9: (20.0, 29.0),  # Near Kitchen Right
            10: (10.0, 15.0),  # Center T-Far
            11: (10.0, 29.0),  # Center T-Near
            12: (10.0, 0.0),  # Mid-Top Baseline
            13: (10.0, 44.0),  # Mid-Bottom Baseline
        }

        self.white_canvas_rectangle_height = 280
        # Calculate width to maintain court aspect ratio (20/44)
        self.white_canvas_rectangle_width = int(
            self.white_canvas_rectangle_height
            * (self.PICKLEBALL_COURT_WIDTH_feet / self.PICKLEBALL_COURT_LENGTH_feet)
        )
        # Calculate kitchen line ratio for drawing
        self.distance_baseline_to_kitchen = (
            self.PICKLEBALL_COURT_LENGTH_feet / 2
        ) - self.PICKLEBALL_KITCHEN_LENGTH_feet
        self.pickleball_kitchen_line_ratio = (
            self.distance_baseline_to_kitchen / self.PICKLEBALL_COURT_LENGTH_feet
        )

        self.buffer = 20  # Buffer around the white canvas to the frame edge
        self.padding = 15  # Padding around the court lines to the edge of white canvas

        self.set_white_canvas_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()

    # -----------------------------------------------------------
    # 1. SETUP METHODS (Positioning and Coordinates) -
    # -----------------------------------------------------------
    def draw_white_canvas(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws a filled white rectangle on the given frame within the specified drawing area.
        Args:
            frame (np.ndarray): The image frame on which to draw the white canvas.
        Returns:
            np.ndarray: The frame with the drawn white canvas.
        """
        cv2.rectangle(
            frame,
            self.drawing_rectangle[0],
            self.drawing_rectangle[1],
            (255, 255, 255),
            cv2.FILLED,
        )
        return frame

    def set_white_canvas_position(self, frame: np.ndarray) -> None:
        """
        Sets the position of the white canvas rectangle on the given video frame.
        Args:
            frame (np.ndarray): The video frame on which to set the white canvas position.
        Returns:
            None
        """
        frame_height, frame_width = frame.shape[:2]
        self.top_left_x = frame_width - self.white_canvas_rectangle_width - self.buffer
        self.top_left_y = self.buffer
        self.bottom_right_x = frame_width - self.buffer
        self.bottom_right_y = self.buffer + self.white_canvas_rectangle_height

        # Define the drawing rectangle coordinates
        self.drawing_rectangle = (
            (self.top_left_x, self.top_left_y),
            (self.bottom_right_x, self.bottom_right_y),
        )

    def set_mini_court_position(self):
        """
        Calculates and sets the top-left and bottom-right coordinates of the mini court.
        Args:
            None
        Returns:
            None
        """
        self.mini_court_top_left = (
            self.top_left_x + self.padding,
            self.top_left_y + self.padding,
        )
        self.mini_court_bottom_right = (
            self.bottom_right_x - self.padding,
            self.bottom_right_y - self.padding,
        )

    def set_court_drawing_key_points(self):
        """
        Calculates and sets the key points and lines required for drawing a mini pickleball court.
        Args:
            None
        Returns:
            None
        """
        # Calculate the dimensions of the mini court drawing area
        court_width = self.mini_court_bottom_right[0] - self.mini_court_top_left[0]
        court_height = self.mini_court_bottom_right[1] - self.mini_court_top_left[1]

        # Calculate Pixels/Foot Ratio for drawing ball/player positions
        self.pixels_per_foot_x = court_width / self.PICKLEBALL_COURT_WIDTH_feet
        self.pixels_per_foot_y = court_height / self.PICKLEBALL_COURT_LENGTH_feet

        # X_left and Y_top define the top-left corner of the drawing area.
        X_left, Y_top = self.mini_court_top_left

        # X_right and Y_bottom define the bottom-right corner of the drawing area.
        X_right, Y_bottom = self.mini_court_bottom_right

        # Midpoints
        X_center = X_left + (X_right - X_left) / 2
        Y_net = Y_top + court_height / 2  # Y coordinate of the net line

        # Kitchen lines (Non-Volley Zone) Y coordinates based on the ratio
        Y_KL_far = (
            Y_top + court_height * self.pickleball_kitchen_line_ratio
        )  # Far side kitchen line Y
        Y_KL_near = (
            Y_bottom - court_height * self.pickleball_kitchen_line_ratio
        )  # Near side kitchen line Y

        # 14 Key Points for the Full Court (in x, y pairs)
        self.drawing_key_points = [
            # 0-3: Outer Boundary (Order: TL -> TR -> BR -> BL)
            X_left,
            Y_top,  # 0: Top-Left (TL) - Far Baseline, Left Sideline
            X_right,
            Y_top,  # 1: Top-Right (TR) - Far Baseline, Right Sideline
            X_right,
            Y_bottom,  # 2: Bottom-Right (BR) - Near Baseline, Right Sideline
            X_left,
            Y_bottom,  # 3: Bottom-Left (BL) - Near Baseline, Left Sideline
            # 4-5: Net Line (Mid-court, full width)
            X_left,
            Y_net,  # 4: Net Left
            X_right,
            Y_net,  # 5: Net Right
            # 6-9: Kitchen Lines (Full width - Non-Volley Zone Boundary)
            X_left,
            Y_KL_far,  # 6: Far Kitchen Line Left
            X_right,
            Y_KL_far,  # 7: Far Kitchen Line Right
            X_left,
            Y_KL_near,  # 8: Near Kitchen Line Left
            X_right,
            Y_KL_near,  # 9: Near Kitchen Line Right
            # 10-11: Center Service Line T-Intersections (marks service boxes)
            X_center,
            Y_KL_far,  # 10: Center T-Far (T-intersection with far kitchen line)
            X_center,
            Y_KL_near,  # 11: Center T-Near (T-intersection with near kitchen line)
            # 12-13: Mid-Baselines (used for drawing the center service line segments)
            X_center,
            Y_top,  # 12: Mid-Top Baseline (Far)
            X_center,
            Y_bottom,  # 13: Mid-Bottom Baseline (Near)
        ]

        # Lines connecting the key points (using indices of drawing_key_points)
        self.lines = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Outer rectangle (TL->TR->BR->BL->TL)
            (4, 5),  # Net Line (Mid-court)
            (6, 7),  # Far Kitchen Line
            (8, 9),  # Near Kitchen Line
            # Center Service Lines (run from baseline to kitchen line in each half)
            (12, 10),  # Top Half: Mid-Top Baseline (12) to Center/Kitchen T-Far (10)
            (
                13,
                11,
            ),  # Bottom Half: Mid-Bottom Baseline (13) to Center/Kitchen T-Near (11)
        ]

    def draw_court(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws the pickleball court lines and highlights the net on the given frame.
        Args:
            frame (np.ndarray): The image frame on which to draw the court.
        Returns:
            np.ndarray: The frame with the drawn court.
        """
        # Draw Lines (Black color)
        for line in self.lines:
            # Get the coordinates for the start point (index * 2 for x, index * 2 + 1 for y)
            start_index = line[0] * 2
            end_index = line[1] * 2

            start_point = (
                int(self.drawing_key_points[start_index]),
                int(self.drawing_key_points[start_index + 1]),
            )
            end_point = (
                int(self.drawing_key_points[end_index]),
                int(self.drawing_key_points[end_index + 1]),
            )
            cv2.line(
                frame, start_point, end_point, (0, 0, 0), 2
            )  # Black lines, thickness 2

        # Highlight the net (e.g., in Blue) - using points 4 and 5
        net_start_point = (
            int(self.drawing_key_points[4 * 2]),
            int(self.drawing_key_points[4 * 2 + 1]),
        )
        net_end_point = (
            int(self.drawing_key_points[5 * 2]),
            int(self.drawing_key_points[5 * 2 + 1]),
        )
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)  # Blue net

        return frame

    # -----------------------------------------------------------
    # 2. COORDINATE CONVERSION CORE LOGIC
    # -----------------------------------------------------------

    def get_mini_court_pixels_from_feet(
        self, real_world_feet_position: Tuple[float, float]
    ) -> Tuple[int, int]:
        """
        Convert an absolute real-world position (in feet) to pixel coordinates on the mini court.
        Args:
            real_world_feet_position (Tuple[float, float]): A tuple (x, y) giving the position in feet
                in real-world court coordinates.
        Returns:
            Tuple[int, int]: A tuple (x_pixel, y_pixel) giving the corresponding pixel coordinates on
            the mini court (returned as integers).
        """
        real_world_x_feet, real_world_y_feet = real_world_feet_position

        mini_court_x_pixel = (
            self.mini_court_top_left[0] + real_world_x_feet * self.pixels_per_foot_x
        )
        mini_court_y_pixel = (
            self.mini_court_top_left[1] + real_world_y_feet * self.pixels_per_foot_y
        )

        return (int(mini_court_x_pixel), int(mini_court_y_pixel))

    def compute_homography_matrix(
        self, court_keypoints_map: Dict
    ) -> Union[np.ndarray, None]:
        """
        Compute a homography matrix that maps image pixel coordinates to real-world court coordinates.
        Args:
            court_keypoints_map (Dict): Mapping from integer keypoint indices to (x, y) pixel coordinates
                detected in the video frame. Expected to contain the required corner/keypoints used for
                homography estimation.
        Returns:
            Union[np.ndarray, None]: A 3x3 homography matrix (dtype float32) that maps image points to
            court coordinates (feet). Returns None if required keypoints are missing or homography
            cannot be computed.
        """
        # Use all key points for better accuracy
        source_points_indices = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        # Ensure all necessary keypoints are present
        if not all(idx in court_keypoints_map for idx in source_points_indices):
            logger.warning(
                "Cannot compute homography: Missing one or more corner keypoints."
            )
            return None

        # Source points from the video frame (in pixels)
        source_points = np.array(
            [court_keypoints_map[i] for i in source_points_indices], dtype=np.float32
        )

        # Destination points in the real world (in feet)
        destination_points = np.array(
            [self.keypoint_feet_mapping[i] for i in source_points_indices],
            dtype=np.float32,
        )

        # Compute the homography matrix. Use Ransac to handle outliers. 10.0 pixel threshold
        homography_matrix, _ = cv2.findHomography(
            source_points, destination_points, cv2.RANSAC, 10.0
        )
        return homography_matrix

    def transform_pixel_to_feet(
        self, pixel_coord: Tuple[int, int], homography_matrix: np.ndarray
    ) -> Union[Tuple[float, float], None]:
        """
        Transform a pixel coordinate from image space into court coordinates (feet)
        using the homography matrix.
        Args:
            pixel_coord (Tuple[int, int]): (x, y) pixel coordinates in the image.
            homography_matrix (np.ndarray): 3x3 homography matrix mapping image
                points to court/world coordinates in feet. If None, no transformation
                is performed.
        Returns:
            Union[Tuple[float, float], None]: A tuple (x_feet, y_feet) giving the
                transformed coordinates in feet, or None if homography_matrix is None.
        """
        if homography_matrix is None:
            return None

        # The point needs to be in a specific format for perspectiveTransform
        point_to_transform = np.array([[pixel_coord]], dtype=np.float32)

        # Apply the perspective transformation
        transformed_point = cv2.perspectiveTransform(
            point_to_transform, homography_matrix
        )

        # Extract the (x, y) coordinates in feet
        return (transformed_point[0][0][0], transformed_point[0][0][1])

    def get_real_world_feet_and_mini_court_coordinates(
        self,
        object_position: Tuple[int, int],
        closest_key_point: Tuple[int, int],
        closest_key_point_index: int,
        player_height_in_pixels: int,
        player_height_in_feet: float,
    ) -> Tuple[Tuple[float, float], Tuple[int, int]]:
        """
        Converts a position from the original video frame to a position on the mini-court.

        Args:
            object_position: The (x, y) position of the object in the original video frame.
            closest_key_point: The (x, y) position of the closest keypoint in the original video frame.
            closest_key_point_index: The index of the closest keypoint.
            player_height_in_pixels: The height of the player in pixels.
            player_height_in_feet: The height of the player in feet.

        Returns:
            Tuple[Tuple[float, float], Tuple[int, int]]:
            (Real-World Feet Position, Mini-Court Pixel Position)
        """
        # Measure the pixel distance (offset) from the object to the original keypoint
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = (
            measure_xy_distance(object_position, closest_key_point)
        )

        # Convert the pixel distance to a real-world distance in feet (offset)
        distance_from_keypoint_x_feet = convert_pixel_distance_to_feet(
            distance_from_keypoint_x_pixels,
            player_height_in_feet,
            player_height_in_pixels,
        )
        distance_from_keypoint_y_feet = convert_pixel_distance_to_feet(
            distance_from_keypoint_y_pixels,
            player_height_in_feet,
            player_height_in_pixels,
        )

        # 1. Calculate the ABSOLUTE real-world feet position
        closest_feet_keypoint = self.keypoint_feet_mapping[closest_key_point_index]

        # The object's absolute position = KP's absolute feet position + calculated feet offset
        real_world_x_feet = closest_feet_keypoint[0] + distance_from_keypoint_x_feet
        real_world_y_feet = closest_feet_keypoint[1] + distance_from_keypoint_y_feet
        real_world_feet_position = (real_world_x_feet, real_world_y_feet)

        # 2. Convert the absolute feet position to mini-court pixels
        mini_court_pixel_position = self.get_mini_court_pixels_from_feet(
            real_world_feet_position
        )

        return real_world_feet_position, mini_court_pixel_position

    def build_kp_map(
        self, court_key_points: List[Dict[str, Union[int, float, None]]]
    ) -> Dict[int, Tuple[float, float]]:
        """
        Build a mapping from keypoint id to (x, y) coordinates.

        This function processes a list of keypoint dictionaries and returns a dictionary
        mapping each valid keypoint id (as int) to a 2-tuple of coordinates (x, y) as floats.
        Entries that are None, missing required fields, or contain None for id or both
        coordinate values are ignored.

        Args:
            court_key_points (List[Dict[str, Union[int, float, None]]]):
                A list of keypoint dictionaries. Each dictionary is expected to include
                at least:
                    - 'kp_id': identifier for the keypoint (int or convertible to int)
                    - 'x' and 'y': coordinates in pixels (float/int) OR
                    - 'x_feet' and 'y_feet': coordinates in feet (float/int)
                Any of the above keys may be missing or set to None; such entries will be skipped.

        Returns:
            Dict[int, Tuple[float, float]]:
                A dictionary mapping keypoint ids (int) to coordinate tuples (x, y) where
                coordinates are floats. Only entries with a present and non-None id and
                at least one valid coordinate source are included. If no valid keypoints are
                found, an empty dict is returned.
        """
        kp_map = {}
        for kp in court_key_points:
            if kp is None:
                continue
            kp_id = kp.get("kp_id")
            x = kp.get("x")
            y = kp.get("y")
            if kp_id is None or x is None or y is None:
                continue
            kp_map[int(kp_id)] = (float(x), float(y))
        return kp_map

    def convert_bounding_boxes_to_mini_court_coordinates(
        self,
        player_boxes: List[Dict[int, List[int]]],
        ball_boxes: List[Dict[int, List[int]]],
        court_key_points: List[Dict],
    ) -> Tuple[
        List[Dict[int, Dict[str, Union[Tuple[int, int], Tuple[float, float]]]]],
        List[Dict[str, Dict[str, Union[Tuple[int, int], Tuple[float, float]]]]],
    ]:
        """
        Converts player and ball bounding boxes from video frame coordinates to both real-world feet positions and mini-court pixel positions.
        For each frame, this method:
          - Determines the foot position of each player and the center position of the ball (if present).
          - Finds the closest court keypoint for each player and the ball.
          - Estimates each player's height in pixels using bounding box heights over a window of frames. Taking reference from players real-world heights.
          - Converts the foot position of each player and the ball to both real-world feet coordinates and mini-court pixel coordinates.
          - Identifies the player closest to the ball and computes the ball's position.
        Args:
            player_boxes (List[Dict[int, List[int]]]):
                A list where each element corresponds to a frame and contains a dictionary mapping player ID to their bounding box coordinates [x1, y1, x2, y2].
            ball_boxes (List[Dict[int, List[int]]]):
                A list where each element corresponds to a frame and contains a dictionary mapping the ball ID (typically 0) to its bounding box coordinates [x1, y1, x2, y2].
            court_key_points (List[Dict]):
                A list of dictionaries representing key points on the court for geometric reference.
        Returns:
            Tuple[
                List[Dict[int, Dict[str, Union[Tuple[int, int], Tuple[float, float]]]]],
                List[Dict[str, Dict[str, Union[Tuple[int, int], Tuple[float, float]]]]]
            ]:
                - output_player_metrics:
                    A list (per frame) of dictionaries mapping player IDs to a dictionary with:
                        - 'mini_court_pos': (x, y) pixel coordinates on the mini-court image.
                        - 'feet_pos': (x, y) real-world coordinates in feet.
                - output_ball_metrics:
                    A list (per frame) of dictionaries with key "ball" mapping to a dictionary with:
                        - 'mini_court_pos': (x, y) pixel coordinates on the mini-court image.
                        - 'ball_pos': (x, y) real-world coordinates in feet.
                    The ball's position is only included for the player closest to the ball in each frame.
        Note:
            - The mapping from bounding box coordinates to real-world and mini-court coordinates relies on court keypoints and player height estimation.
        """

        # This list will store tuples of (mini_court_pos, feet_pos)
        output_player_metrics = []
        output_ball_metrics = []

        # Change to (x1,y1,x2,y2...) format for keypoints
        court_keypoints_map = self.build_kp_map(court_key_points)

        # Compute the homography matrix for coordinate transformation
        homography_matrix = self.compute_homography_matrix(court_keypoints_map)

        # Use the center line keypoints for reference to calculate distances
        court_key_points_ref_distance = [12, 10, 11, 13]

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box_dict = ball_boxes[frame_num]

            # --- Ball Position and Closest Player ID ---
            if ball_box_dict and 0 in ball_box_dict and player_bbox:

                # Get the ball's position and the closest player to the ball
                ball_box = ball_box_dict[0]
                ball_position = get_center_of_bbox(ball_box)
                closest_player_id_to_ball = min(
                    player_bbox.keys(),
                    key=lambda player_id: measure_distance(
                        ball_position, get_foot_position(player_bbox[player_id])
                    ),
                )
            else:
                ball_position = None
                closest_player_id_to_ball = None

            current_frame_player_metrics = {}
            current_frame_ball_metrics = {}

            for player_id, bbox in player_bbox.items():

                # Get foot position and closest keypoint for the player
                foot_position = get_foot_position(bbox)
                closest_kp_index = get_closest_keypoint_index(
                    foot_position, court_keypoints_map, court_key_points_ref_distance
                )
                closest_kp_coordinates = court_keypoints_map[closest_kp_index]

                # Estimate player height in pixels using a window of frames around the current frame
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 30)

                # Gather heights from bounding boxes in the frame window. This will manage the players height in pixels when they bend down
                player_bbox_height_pixels = [
                    get_height_of_bbox(player_boxes[i][player_id])
                    for i in range(frame_index_min, frame_index_max)
                    if player_id in player_boxes[i]
                ]
                player_height_pixels = (
                    max(player_bbox_height_pixels) if player_bbox_height_pixels else 0
                )

                # Get player's real-world height in feet (default to 5.8 ft if unknown)
                player_height_feet = self.PLAYER_HEIGHTS_FEET.get(player_id, 5.8)

                # Returns both the real-world feet position and the mini-court pixel position.
                real_world_feet_position, mini_court_player_position = (
                    self.get_real_world_feet_and_mini_court_coordinates(
                        object_position=foot_position,
                        closest_key_point=closest_kp_coordinates,
                        closest_key_point_index=closest_kp_index,
                        player_height_in_pixels=player_height_pixels,
                        player_height_in_feet=player_height_feet,
                    )
                )

                # Store both mini_court_pos (for drawing) and real_world_feet_pos (for metrics)
                current_frame_player_metrics[player_id] = {
                    "mini_court_pos": mini_court_player_position,
                    "feet_pos": real_world_feet_position,
                }

                # Ball's position is only calculated once per frame, using the closest player as a reference.
                if closest_player_id_to_ball == player_id and ball_position is not None:

                    # Convert ball_position to integers for pixel-based functions
                    ball_pos_int = (int(ball_position[0]), int(ball_position[1]))

                    # Use Homography to get the ball's real-world position
                    real_world_ball_position = self.transform_pixel_to_feet(
                        ball_pos_int, homography_matrix
                    )

                    if real_world_ball_position:
                        # Convert the real-world feet position to mini-court pixel position
                        mini_court_ball_position = self.get_mini_court_pixels_from_feet(
                            real_world_ball_position
                        )

                        current_frame_ball_metrics = {
                            "ball": {
                                "mini_court_pos": mini_court_ball_position,
                                "ball_pos": real_world_ball_position,
                            }
                        }

            output_ball_metrics.append(current_frame_ball_metrics)
            output_player_metrics.append(current_frame_player_metrics)

        return output_player_metrics, output_ball_metrics

    # -----------------------------------------------------------
    # 3. REAL-WORLD METRICS UTILITY METHODS
    # -----------------------------------------------------------

    def get_court_feet_distance_between_points(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """
        Calculates the Euclidean distance in feet between two points on the court.
        Args:
            point1 (Tuple[float, float]): The (x, y) coordinates of the first point in feet.
            point2 (Tuple[float, float]): The (x, y) coordinates of the second point in feet.
        Returns:
            float: The Euclidean distance between the two points in feet.
        """

        p1 = np.array(point1)
        p2 = np.array(point2)

        # Euclidean distance
        distance_feet = np.linalg.norm(p1 - p2)
        return float(distance_feet)

    def calculate_speed_kph(self, distance_feet: float, frame_rate: int) -> float:
        """
        Calculates speed in Kilometers Per Hour (kph).

        Args:
            distance_feet (float): Distance covered in feet over one frame.
            frame_rate (int): Frames per second (FPS) of the video.

        Returns:
            float: Speed in km/h.
        """
        # Convert distance from feet to meters
        distance_meters = distance_feet * self.FEET_TO_METERS

        # Time taken for one frame (in seconds)
        time_seconds = 1.0 / frame_rate

        # Speed in meters per second (m/s)
        speed_mps = distance_meters / time_seconds

        # Convert m/s to km/h
        speed_kph = speed_mps * self.KPH_FACTOR

        return speed_kph

    # -----------------------------------------------------------
    # 4. Drawing the objects on the mini-court
    # -----------------------------------------------------------

    def draw_points_on_mini_court(
        self,
        frames: list[np.ndarray],
        metric_data: list[dict],
        color: tuple = (255, 0, 0),
    ) -> list[np.ndarray]:
        """
        Draws points (Players and Ball) and object IDs on each frame of a mini court visualization.
        Args:
            frames (list of np.ndarray): List of image frames to draw on.
            metric_data (list of dict): List where each element corresponds to a frame and contains a dictionary mapping object IDs to their metrics, including 'mini_court_pos' (tuple of x, y coordinates).
            color (tuple, optional): BGR color tuple for drawing points and text. Defaults to (255, 0, 0).
        Returns:
            list of np.ndarray: The list of frames with points and object IDs drawn.
        """

        for frame_num, frame in enumerate(frames):
            for object_id, metrics in metric_data[frame_num].items():

                # We only need the mini_court_pos for drawing
                if "mini_court_pos" in metrics:
                    x, y = metrics["mini_court_pos"]
                    x = int(x)
                    y = int(y)
                    cv2.circle(frame, (x, y), 5, color, -1)

                    # Add ID label for clarity
                    cv2.putText(
                        frame,
                        str(object_id),
                        (x + 8, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
        return frames

    def draw_mini_court_on_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Draws a mini-court and its keypoints on each frame in the provided list.
        For each frame, this method:
            - Draws a white canvas over the frame.
            - Draws the court layout.
            - Draws 14 mini-court keypoints as red circles and labels them with their indices. (This part is commented out for development purposes.)
        Args:
            frames (list[np.ndarray]): List of image frames (as NumPy arrays) to process.
        Returns:
            list[np.ndarray]: List of frames with the mini-court and keypoints drawn.
        """

        output_frames = []
        for i, frame in enumerate(frames):
            frame = self.draw_white_canvas(frame)
            frame = self.draw_court(frame)

            # Draw mini-court keypoints as red circles and label them. Used for development purpose
            # for kp_idx in range(14):
            #     x = int(self.drawing_key_points[kp_idx * 2])
            #     y = int(self.drawing_key_points[kp_idx * 2 + 1])
            #     cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red, filled
            #     cv2.putText(
            #         frame,
            #         str(kp_idx),
            #         (x + 8, y - 8),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.4,
            #         (0, 0, 255),
            #         1,
            #         cv2.LINE_AA
            #     )

            output_frames.append(frame)
        return output_frames
