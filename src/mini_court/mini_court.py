import cv2
import numpy as np

class MiniCourt:
    def __init__(self, frame: np.ndarray):

        self.white_canvas_rectangle_height = 280
        # Pickleball court dimensions: 20 ft wide x 44 ft long
        PICKLEBALL_COURT_WIDTH = 20
        PICKLEBALL_COURT_LENGTH = 44
        # No Volley Zone (Kitchen) dimensions: 7 ft from each side of the net
        PICKLEBALL_KITCHEN_LENGTH = 7
        # Pickleball net height: 36 inches (3 ft)
        PICKLEBALL_NET_HEIGHT = 3

        distance_baseline_to_kitchen = (PICKLEBALL_COURT_LENGTH / 2) - PICKLEBALL_KITCHEN_LENGTH
        self.pickleball_kitchen_line_ratio = distance_baseline_to_kitchen / PICKLEBALL_COURT_LENGTH

        self.white_canvas_rectangle_width = int(self.white_canvas_rectangle_height * (PICKLEBALL_COURT_WIDTH / PICKLEBALL_COURT_LENGTH))
        self.buffer = 20 # Buffer around the white canvas to the frame edge
        self.padding = 15 # Padding around the court lines to the edge of white canvas

        self.set_white_canvas_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()

    def draw_white_canvas(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws the defined white canvas rectangle on the given video frame.
        Args:
            frame (numpy.ndarray): The input video frame to draw on.
        Returns:
            numpy.ndarray: The video frame with the drawn rectangle.
        """
        cv2.rectangle(frame, self.drawing_rectangle[0], self.drawing_rectangle[1], (255, 255, 255), cv2.FILLED)  # Draw rectangle in white color
        return frame

    def set_white_canvas_position(self, frame: np.ndarray) -> None:
        """
        Sets the position of the drawing rectangle in the given video frame.
        Args:
            frame (numpy.ndarray): The input video frame.
        """
        frame_height, frame_width = frame.shape[:2]
        self.top_left_x = frame_width - self.white_canvas_rectangle_width - self.buffer
        self.top_left_y = self.buffer
        self.bottom_right_x = frame_width - self.buffer
        self.bottom_right_y = self.buffer + self.white_canvas_rectangle_height

        # Define the drawing rectangle coordinates
        self.drawing_rectangle = (
            (self.top_left_x, self.top_left_y),
            (self.bottom_right_x, self.bottom_right_y)
        )

    def set_mini_court_position(self):
        """
        Sets the position of the mini-court within the drawing rectangle.
        """
        self.mini_court_top_left = (
            self.top_left_x + self.padding,
            self.top_left_y + self.padding
        )
        self.mini_court_bottom_right = (
            self.bottom_right_x - self.padding,
            self.bottom_right_y - self.padding
        )

    def set_court_drawing_key_points(self):
        """
        Sets the drawing key points for the mini-court based on accurate Pickleball court
        proportions derived from the app.yaml dimensions.

        The points are now ordered starting from the near-side (bottom) corners
        to match common court detection schemes.
        """
        # Calculate the dimensions of the mini court drawing area
        court_height = self.mini_court_bottom_right[1] - self.mini_court_top_left[1]

        # Coordinates for easier reading
        # X_left and Y_top define the top-left corner of the drawing area.
        X_left, Y_top = self.mini_court_top_left
        # X_right and Y_bottom define the bottom-right corner of the drawing area.
        X_right, Y_bottom = self.mini_court_bottom_right

        # Midpoints
        X_center = X_left + (X_right - X_left) / 2
        Y_net = Y_top + court_height / 2 # Y coordinate of the net line

        # Kitchen lines (Non-Volley Zone) Y coordinates based on the ratio
        # (distance from baseline / total length)
        Y_KL_far = Y_top + court_height * self.pickleball_kitchen_line_ratio # Far side kitchen line Y
        Y_KL_near = Y_bottom - court_height * self.pickleball_kitchen_line_ratio # Near side kitchen line Y

        # 14 Key Points for the Full Court (in x, y pairs)
        self.drawing_key_points = [
            # 0-3: Outer Boundary (Order: BL -> BR -> TR -> TL)
            X_left, Y_bottom, # 0: Bottom-Left (BL) - Near Baseline, Left Sideline
            X_right, Y_bottom, # 1: Bottom-Right (BR) - Near Baseline, Right Sideline
            X_right, Y_top, # 2: Top-Right (TR) - Far Baseline, Right Sideline
            X_left, Y_top, # 3: Top-Left (TL) - Far Baseline, Left Sideline

            # 4-5: Net Line (Mid-court, full width)
            X_left, Y_net, # 4: Net Left
            X_right, Y_net, # 5: Net Right

            # 6-9: Kitchen Lines (Full width - Non-Volley Zone Boundary)
            X_left, Y_KL_far, # 6: Far Kitchen Line Left
            X_right, Y_KL_far, # 7: Far Kitchen Line Right
            X_left, Y_KL_near, # 8: Near Kitchen Line Left
            X_right, Y_KL_near, # 9: Near Kitchen Line Right

            # 10-11: Center Service Line T-Intersections (marks service boxes)
            X_center, Y_KL_far, # 10: Center T-Far (T-intersection with far kitchen line)
            X_center, Y_KL_near, # 11: Center T-Near (T-intersection with near kitchen line)

            # 12-13: Mid-Baselines (used for drawing the center service line segments)
            X_center, Y_top, # 12: Mid-Top Baseline (Far)
            X_center, Y_bottom, # 13: Mid-Bottom Baseline (Near)
        ]

        # Lines connecting the key points (using indices of drawing_key_points)
        # Note: Each point index represents an (x, y) pair in the list.
        self.lines = [
            (0, 1), (1, 2), (2, 3), (3, 0), # Outer rectangle (BL->BR->TR->TL->BL)
            (4, 5), # Net Line (Mid-court)
            (6, 7), # Far Kitchen Line
            (8, 9), # Near Kitchen Line

            # Center Service Lines (run from baseline to kitchen line in each half)
            (12, 10), # Top Half: Mid-Top Baseline (12) to Center/Kitchen T-Far (10)
            (13, 11)  # Bottom Half: Mid-Bottom Baseline (13) to Center/Kitchen T-Near (11)
        ]

    def draw_court(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws the mini court lines onto the white canvas area of the frame.
        """
        # Draw Lines (Black color)
        for line in self.lines:
            # Get the coordinates for the start point (index * 2 for x, index * 2 + 1 for y)
            start_index = line[0] * 2
            end_index = line[1] * 2

            start_point = (
                int(self.drawing_key_points[start_index]),
                int(self.drawing_key_points[start_index + 1])
            )
            end_point = (
                int(self.drawing_key_points[end_index]),
                int(self.drawing_key_points[end_index + 1])
            )
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2) # Black lines, thickness 2

        # Highlight the net (e.g., in Red) - using points 4 and 5
        # The indices are multiplied by 2 because drawing_key_points stores x and y consecutively
        net_start_point = (int(self.drawing_key_points[4*2]), int(self.drawing_key_points[4*2+1]))
        net_end_point = (int(self.drawing_key_points[5*2]), int(self.drawing_key_points[5*2+1]))
        cv2.line(frame, net_start_point, net_end_point, (0, 0, 255), 2) # Red net

        return frame

    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_white_canvas(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames


