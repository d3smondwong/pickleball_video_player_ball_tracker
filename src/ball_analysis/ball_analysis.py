import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class BallAnalysis:
    """
    Analyzes player and ball detections to determine positions on the 2D court plane
    (mini-court view) and detect key events like hits and bounces.
    """

    def __init__(self, court_keypoints: Dict[str, Tuple[int, int]]):
        """
        Initializes the BallAnalysis with court keypoints to establish the
        perspective transformation (Homography).

        Args:
            court_keypoints: A dictionary of court keypoint names and their
                             (x, y) pixel coordinates in the video frame.
        """
        self.court_keypoints = court_keypoints

        # Real-world court dimensions for the destination plane (in feet)
        # Assuming origin (0,0) is the far-left corner (top-left in mini-court view)
        self.COURT_WIDTH_FEET = 20.0
        self.COURT_LENGTH_FEET = 44.0
        self.H = self._calculate_homography_matrix()
    def _calculate_homography_matrix(self) -> np.ndarray:
        """
        Calculates the Homography matrix (H) for perspective transformation.
        This maps 2D video coordinates to 2D court coordinates (in feet).
        """
        # Define source points (video pixel coordinates) from the keypoints
        # We need at least 4 keypoints for homography calculation.
        # Standard court points used: Far Baseline (Left & Right), Near Baseline (Left & Right).

        # Note: This is a simplified selection. A robust model would use
        # all available keypoints for a more accurate perspective transform.
        try:
            # Source points (4 keypoints from video frame, accessed by keypoint ID)
            src_pts = np.array([
                [self.court_keypoints[0]['x'], self.court_keypoints[0]['y']],    # kp_id 0: Far Baseline Left
                [self.court_keypoints[1]['x'], self.court_keypoints[1]['y']],    # kp_id 1: Far Baseline Right
                [self.court_keypoints[2]['x'], self.court_keypoints[2]['y']],    # kp_id 2: Near Baseline Right
                [self.court_keypoints[3]['x'], self.court_keypoints[3]['y']]     # kp_id 3: Near Baseline Left
            ], dtype=np.float32)

            # Destination points (corresponding real-world coordinates in feet)
            # Origin (0, 0) is typically the far-left corner.
            dst_pts = np.array([
                [0.0, 0.0],
                [float(self.COURT_WIDTH_FEET), 0.0],
                [float(self.COURT_WIDTH_FEET), float(self.COURT_LENGTH_FEET)],
                [0.0, float(self.COURT_LENGTH_FEET)]
            ], dtype=np.float32)

            # --- Diagnostic Logging ---
            logger.info(f"Source Points (Video Pixels): \n{src_pts}")
            logger.info(f"Destination Points (Court Feet): \n{dst_pts}")

            # Calculate the homography matrix
            H, _ = cv2.findHomography(src_pts, dst_pts)
            return H
        except Exception as e:
            logger.error(f"Error calculating homography matrix. Using identity matrix as fallback: {e}")
            # Fallback identity matrix - this will yield incorrect results but prevents crashing
            return np.identity(3)

    def _map_to_mini_court_coords(self, video_coords: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        Maps a single (x, y) video pixel coordinate to a real-world (x, y) foot coordinate
        on the court plane using the calculated homography matrix H.

        Args:
            video_coords: (x, y) pixel coordinate from the video frame.

        Returns:
            (x_feet, y_feet) coordinate on the court, or None if transformation fails.
        """
        if self.H is None:
            return None

        # Convert to the format required for perspectiveTransform: [ [[x, y]] ]
        point = np.array([[[video_coords[0], video_coords[1]]]], dtype='float32')

        try:
            # Apply the perspective transformation
            transformed_point = cv2.perspectiveTransform(point, self.H)

            # The result is in feet on the court plane
            x_feet = float(transformed_point[0, 0, 0])
            y_feet = float(transformed_point[0, 0, 1])

            return (x_feet, y_feet)

        except Exception as e:
            logger.warning(f"Failed to transform coordinate {video_coords}: {e}")
            return None

    def _detect_hits(self, frame_idx: int, ball_pos: Optional[Tuple[float, float]], player_poses: Dict[int, Tuple[float, float]]) -> Optional[Dict[str, Any]]:
        """
        MOCK LOGIC: Detects if a hit occurred in the current frame.
        In a real scenario, this would involve checking ball velocity changes and
        proximity to a player's paddle/body.

        Returns:
            A dictionary with hit details (player_id, shot_type) or None.
        """
        # Mock logic: return a hit every 50 frames
        if frame_idx % 50 == 0 and ball_pos is not None and player_poses:
            # Simple assumption: Player 1 hits first, then Player 2
            player_id = 1 if (frame_idx // 50) % 2 == 0 else 2
            shot_type = "Drive" if np.random.rand() > 0.5 else "Dink"

            return {
                'player_id': player_id,
                'shot_type': shot_type
            }
        return None

    def _detect_bounces(self, frame_idx: int, ball_pos: Optional[Tuple[float, float]]) -> Optional[str]:
        """
        MOCK LOGIC: Detects if the ball bounced on the court.
        In a real scenario, this involves analyzing the vertical trajectory of the ball.

        Returns:
            A string ('In' or 'Out') if a bounce is detected, or None.
        """
        # Mock logic: return a bounce every 30 frames
        if frame_idx % 30 == 0 and ball_pos is not None:
            # Simple assumption: 90% of bounces are "In"
            return "In" if np.random.rand() < 0.9 else "Out"
        return None

    def _get_player_court_location_name(self, y_feet: float) -> str:
        """Determines if a player is at the Baseline, Mid-court, or NVZ (Kitchen)."""

        # NVZ (Kitchen) is 0-7 feet and 37-44 feet from the far baseline (0-44 scale)
        NVZ_LENGTH = 7.0

        if (y_feet >= 0 and y_feet <= NVZ_LENGTH) or \
           (y_feet >= self.COURT_LENGTH_FEET - NVZ_LENGTH and y_feet <= self.COURT_LENGTH_FEET):
            return "NVZ (Kitchen)"
        elif y_feet > NVZ_LENGTH and y_feet < self.COURT_LENGTH_FEET - NVZ_LENGTH:
            return "Mid-court/Transition"
        else:
            # This handles positions exactly on the baseline/net, treating them as close to mid/baseline
            return "Baseline Area"


    def analyze_video(self, player_detections: List[Dict[int, Any]], ball_detections: List[Optional[Tuple[int, int, int, int]]]) -> List[Dict[str, Any]]:
        """
        Performs analysis across all frames.

        Args:
            player_detections: List of player detection results per frame.
            ball_detections: List of ball detection bounding boxes (x, y, w, h) per frame.

        Returns:
            A list of analysis dictionaries, one for each frame.
        """
        analysis_results = []
        num_frames = len(player_detections)

        if num_frames != len(ball_detections):
             logger.error("Player and ball detections list lengths do not match.")
             return []

        for frame_idx in range(num_frames):
            frame_analysis: Dict[str, Any] = {
                'frame_idx': frame_idx,
                'player_mini_court_pos': {},  # {player_id: (x_feet, y_feet)}
                'ball_mini_court_pos': None,
                'player_court_pos': {}, # {player_id: 'NVZ', 'Baseline', etc}
                'hit_info': None,
                'bounce_info': None
            }

            # --- 1. Process Player Positions ---

            # player_detections[frame_idx] is a dictionary of {player_id: (x, y, w, h, confidence)}
            current_player_detections = player_detections[frame_idx]
            player_mini_court_pos = {}
            player_court_pos_names = {}

            for player_id, detection in current_player_detections.items():
                # Use the bottom center of the bounding box as the player's ground contact point
                x_center = int(detection[0] + detection[2] / 2) # x + w/2
                y_bottom = int(detection[1] + detection[3])     # y + h

                mini_court_pos = self._map_to_mini_court_coords((x_center, y_bottom))

                if mini_court_pos:
                    player_mini_court_pos[player_id] = mini_court_pos
                    x_feet, y_feet = mini_court_pos
                    player_court_pos_names[player_id] = self._get_player_court_location_name(y_feet)

            frame_analysis['player_mini_court_pos'] = player_mini_court_pos
            frame_analysis['player_court_pos'] = player_court_pos_names

            # --- 2. Process Ball Position ---

            # ball_detections[frame_idx] is a list of detections, typically one, e.g., [(x, y, w, h)]
            current_ball_detections = ball_detections[frame_idx]

            if current_ball_detections:
                # Use the center of the first ball detection's bounding box
                ball_bbox = current_ball_detections[0] # Assuming only one ball
                x_center = int(ball_bbox[0] + ball_bbox[2] / 2)
                y_center = int(ball_bbox[1] + ball_bbox[3] / 2)

                ball_mini_court_pos = self._map_to_mini_court_coords((x_center, y_center))
                frame_analysis['ball_mini_court_pos'] = ball_mini_court_pos

            # --- 3. Detect Game Events ---

            ball_pos_feet = frame_analysis['ball_mini_court_pos']

            # Note: Hits and Bounces detection would typically require trajectory history
            # (i.e., comparing pos/velocity across multiple frames), but mock logic is used here.

            frame_analysis['hit_info'] = self._detect_hits(frame_idx, ball_pos_feet, player_mini_court_pos)
            frame_analysis['bounce_info'] = self._detect_bounces(frame_idx, ball_pos_feet)

            analysis_results.append(frame_analysis)

        return analysis_results

if __name__ == "__main__":
    # Example usage for testing (requires mock court keypoints)
    print("BallAnalysis class is defined. Run app.py to use it in the full video pipeline.")
