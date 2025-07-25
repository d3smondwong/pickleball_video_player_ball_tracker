from src.utils.video_utils import read_video, save_video
from src.trackers import PlayerTracker, BallTracker
from src.court_line_detector.court_line_detector import CourtLineDetector
from pathlib import Path
import os
import pickle
from dotenv import load_dotenv
import cv2

def main():
    """
    Main function to process a video file.
    This function performs the following steps:
    1. Validates the existence and format of the input video file.
    2. Reads video frames from the specified input video file.
    3. Ensures the output directory exists or creates it if necessary.
    4. Saves the read video frames to the specified output video file.
    Raises:
        FileNotFoundError: If the input video file does not exist.
        ValueError: If the input path is not a file or has an unsupported format.
    """

    ###
    # Read the frames from the input video file
    ###

    # Ensure the input and output paths are correct
    input_video_path = Path("input_videos/pickleball_highlights.mp4")

    # Checkers
    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")
    if not input_video_path.is_file():
        raise ValueError(f"Input path is not a file: {input_video_path}")
    if not input_video_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        raise ValueError(f"Unsupported video file format: {input_video_path.suffix}. Supported formats are .mp4, .avi, .mov.")

    # Read video frames
    print(f"Reading video frames from: {input_video_path}")
    video_frames = read_video(str(input_video_path))

    ###
    # Track the player using yolo12x
    ###

    # Load model and initiate PlayerTracker to track players for different frames
    model_path = Path('artifacts/models/yolo12x.pt')
    player_tracker = PlayerTracker(model_path=str(model_path))

    player_detections_stub_path = f"artifacts/tracker_stubs/player_detections_{input_video_path.stem}.pkl"

    # Save players detected into a Pickle file. To prevent multiple processing in production
    # Check if the pickle file exists, if not, it will be created
    if not Path(player_detections_stub_path).exists():
        print(f"Stub file not found, it will be created: {player_detections_stub_path}")
        read_from_stub = False
    else:
        print(f"Loading player detections from stub file: {player_detections_stub_path}")
        read_from_stub = True

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=read_from_stub,
                                                     stub_path=player_detections_stub_path
                                                     )

    ###
    # Track the ball using model trained on Roboflow
    ###

    # Load model and initiate BallTracker to track ball for different frames
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if api_key is None:
        raise ValueError("ROBOFLOW_API_KEY environment variable is not set.")
    model_id = "pickleball-vision/6"

    ball_tracker = BallTracker(api_key=api_key, model_id=model_id)

    ball_detections_stub_path = f"artifacts/tracker_stubs/ball_detections_{input_video_path.stem}.pkl"

    # Save balls detection into a Pickle file. To prevent multiple processing in production
    # Check if the pickle file exists, if not, it will be created
    if not Path(ball_detections_stub_path).exists():
        print(f"Stub file not found, it will be created: {ball_detections_stub_path}")
        read_from_stub = False
    else:
        print(f"Loading ball detections from stub file: {ball_detections_stub_path}")
        read_from_stub = True

    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=read_from_stub,
                                                 stub_path=ball_detections_stub_path
                                                 )

    # To interpolate the ball positions when it is not detected in some frames. Does not work well as there are times it goes out of the camera during lobs
    # ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    ###
    # Detect the court lines using the CourtLineDetector
    ###
    """
    # The keypoints model does not work too well yet. To rectify

    court_model_path = "artifacts/models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(model_path=court_model_path)

    """
    # Using Roboflow model for court keypoints detection
    model_id = "pickle-court-keypoints-nluo7-8nk97/4"
    court_line_detector = CourtLineDetector(api_key=api_key, model_id=model_id)
    court_keypoints = court_line_detector.predict_roboflow(video_frames[0])

    ###
    # Draw bounding boxes on the output video frames
    ###
    # Draw bounding boxes on the video frames using the player and ball detections
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    output_video_frames= ball_tracker.draw_bounding_boxes(output_video_frames, ball_detections)

    #"""
    # The keypoints model does not work too well yet.

    # Draw the court keypoints on the output video frames
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    #"""
    ###
    # Save video frames to the output video file
    ###
    output_video_path = Path(f"output_videos/{input_video_path.stem}_output.avi")
    print(f"Saving video frames to: {output_video_path}")

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Check if the parent directory of the output video path exists, if not, create it
    if not output_video_path.parent.exists():
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the video frames to the output video file
    save_video(output_video_frames, str(output_video_path))

if __name__ == "__main__":

    # python -m src.app
    main()
    print("Video processing completed successfully.")