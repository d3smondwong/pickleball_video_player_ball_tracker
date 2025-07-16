from src.utils.video_utils import read_video, save_video
from src.trackers.player_tracker import PlayerTracker
from pathlib import Path

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
    input_video_path = Path("input_videos/input_video.mp4")

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
    player_detections = player_tracker.detect_frames(video_frames)

    ###
    # Draw bounding boxes on the output video frames
    ###

    # Draw bounding boxes on the video frames using the player detections
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)

    ###
    # Save video frames to the output video file
    ###
    output_video_path = Path("output_videos/output_video.avi")
    print(f"Saving video frames to: {output_video_path}")

    # Check if the parent directory of the output video path exists, if not, create it
    if not output_video_path.parent.exists():
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the video frames to the output video file
    save_video(output_video_frames, str(output_video_path))

if __name__ == "__main__":

    # python -m src.app
    main()
    print("Video processing completed successfully.")