from src.utils.video_utils import read_video, save_video
from pathlib import Path

def main():
    # Read video frames from the input video file and save them to the output video file

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

    output_video_path = Path("output_videos/output_video.avi")
    print(f"Saving video frames to: {output_video_path}")

    # Check if the parent directory of the output video path exists, if not, create it
    if not output_video_path.parent.exists():
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the video frames to the output video file
    save_video(video_frames, str(output_video_path))

if __name__ == "__main__":

    # python -m src.app
    main()
    print("Video processing completed successfully.")