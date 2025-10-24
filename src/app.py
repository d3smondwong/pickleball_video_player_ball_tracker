import os
import pickle
import cv2
import json
import hydra
import logging

from src.mini_court.mini_court import MiniCourt
from src.utils.video_utils import read_video, save_video
from src.trackers import PlayerTracker, BallTracker
from src.court_line_detector.court_line_detector import CourtLineDetector
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

@hydra.main(config_path="../config", config_name="app.yaml", version_base="1.2")
def main(cfg: DictConfig):
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

    # Set up logging
    logger = logging.getLogger(os.path.basename(__file__))

    ###
    # Read the frames from the input video file
    ###

    # Ensure the input and output paths are correct
    input_dir = cfg.videos.input_video_folder
    video_name = cfg.videos.video_filename
    input_video_path = Path(input_dir) / video_name

    # Checkers
    if not input_video_path.exists():
        logger.error(f"Input video file not found: {input_video_path}")
    if not input_video_path.is_file():
        logger.error(f"Input path is not a file: {input_video_path}")
    if not input_video_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        logger.error(f"Unsupported video file format: {input_video_path.suffix}. Supported formats are .mp4, .avi, .mov.")

    # Read video frames
    logger.info(f"Reading video frames from: {input_video_path}")
    video_frames = read_video(str(input_video_path))

    ###
    # Track the player using yolo12x
    ###

    # Load model and initiate PlayerTracker to track players for different frames
    model_folder = cfg.models.model_folder
    model_filename = cfg.models.yolo12x
    model_path = Path(model_folder) / model_filename
    player_tracker = PlayerTracker(model_path=str(model_path), cfg=cfg)

    stub_folder = cfg.stubs.stub_folder

    player_detections_stub_path = Path(stub_folder) / cfg.stubs.player_detections_stub.format(video_filename_stem=input_video_path.stem)

    # Save players detected into a Pickle file. To prevent multiple processing in production
    # Check if the pickle file exists, if not, it will be created
    if not Path(player_detections_stub_path).exists():
        logger.info(f"Stub file not found, it will be created: {player_detections_stub_path}")
        read_from_stub = False
    else:
        logger.info(f"Loading player detections from stub file: {player_detections_stub_path}")
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
    model_id = cfg.models.roboflow_ball_model_id

    ball_tracker = BallTracker(api_key=api_key, model_id=model_id)

    ball_detections_stub_path = Path(stub_folder) / cfg.stubs.ball_detections_stub.format(video_filename_stem=input_video_path.stem)

    # Save balls detection into a Pickle file. To prevent multiple processing in production
    # Check if the pickle file exists, if not, it will be created
    if not Path(ball_detections_stub_path).exists():
        logger.info(f"Stub file not found, it will be created: {ball_detections_stub_path}")
        read_from_stub = False
    else:
        logger.info(f"Loading ball detections from stub file: {ball_detections_stub_path}")
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
    # The custom keypoints model does not work too well yet. Trying to find a better model. Using Roboflow model for now

    court_model_path = "artifacts/models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(model_path=court_model_path)

    # Using Roboflow model for court keypoints detection
    model_id = "pickle-court-keypoints-nluo7-8nk97/4"
    court_line_detector = CourtLineDetector(api_key=api_key, model_id=model_id)
    court_keypoints = court_line_detector.predict_roboflow(video_frames[0])
    print(f'court keypoints: {court_keypoints}')
    """
    # Court keypoints calculated manually for now as both roboflow and custom trained model are not very accurate

    court_keypoints_folder = cfg.court_keypoints.court_keypoints_folder
    court_keypoints_filename = cfg.court_keypoints.court_keypoints_filename.format(video_filename_stem=input_video_path.stem)
    logger.info(f'Loading court keypoints from: {court_keypoints_filename}')
    court_keypoints_path = Path(court_keypoints_folder) / court_keypoints_filename

    if not court_keypoints_path.exists():
        raise FileNotFoundError(f"Court keypoints file not found: {court_keypoints_path}")
    with court_keypoints_path.open("r", encoding="utf-8") as f:
        court_keypoints = json.load(f)

    # Filter the player detections to only include the players on the court
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    ###
    # Minicourt
    ###
    mini_court = MiniCourt(video_frames[0], cfg)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                                           ball_detections,
                                                                                                                           court_keypoints)

    ###
    # Annotate the output video frames
    ###
    # Draw bounding boxes on the video frames using the player and ball detections
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    output_video_frames= ball_tracker.draw_bounding_boxes(output_video_frames, ball_detections)

    # Draw the court keypoints on the output video frames. Used only for development purposes
    # court_line_detector = CourtLineDetector()
    # output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court_on_frames(output_video_frames)

    # Draw the player and ball positions on the mini court
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections, color=(0, 255, 0))
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 0, 255))

    # Annotate the main video frames with the frame number. Used only for development purposes
    # final_annotated_frames = []
    # for frame_idx, frame in enumerate(output_video_frames):
    #     # Frame number stays on top left
    #     cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #     final_annotated_frames.append(frame)

    # output_video_frames = final_annotated_frames

    ###
    # Save video frames to the output video file
    ###
    output_video_folder = cfg.videos.output_video_folder
    output_video_path = Path(output_video_folder) / f"{input_video_path.stem}_output.avi"
    logger.info(f"Saving video frames to: {output_video_path}")

    # Check if the parent directory of the output video path exists, if not, create it
    if not output_video_path.parent.exists():
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the video frames to the output video file
    save_video(output_video_frames, str(output_video_path))

if __name__ == "__main__":

    # python -m src.app
    main()
    print("Video processing completed successfully.")