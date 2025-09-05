# Pickleball Player and Ball Tracker

This application analyzes pickleball videos to automatically track player movements and ball trajectories using computer vision. It leverages the YOLOv12x model for player detection and a custom ball tracking model trained on Roboflow.

### Use Case

Accurate tracking of players and the ball enables detailed analysis of individual performance and match dynamics. With reliable detection of court keypoints, the application can generate actionable insights and visualizations to help players improve their gameplay and strategy.

**Current Limitation:** Court keypoint detection is not yet robust, resulting in less accurate distance and velocity calculations. We will remove this feature for now.

**Use Case example**

![Pickleball Player and Ball Tracker Demo](demo_assets/pickleball_highlights_output.gif)

## Contents
- Folder Structure
- Core Functionality
- How to run this application?
- How to use the application?

### Folder Structure
```
.
├── .env
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── demo_assets
|   └── pickleball_highlights_output.gif
├── artifacts
│   ├── models
│   └── tracker_stubs
├── input_videos
├── output_videos
├── runs
├── src
│   ├── court_line_detector
│   │   ├── __init__.py
│   │   └── court_line_detector.py
│   ├── trackers
│   │   ├── __init__.py
│   │   ├── ball_tracker.py
│   │   └── player_tracker.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── bbox_utils.py
│   │   └── video_utils.py
│   ├── app.py
│   └── yolo_inference.py
└── training
    ├── training_data
    ├── model_training_ball_detection.ipynb
    ├── model_training_keypoints_detection.ipynb
    └── model_training_ball.py

```
### Core Functionality

1. Video Frame Extraction

    Reads frames from an input video file and prepares them for analysis.

2. Detection:

    `player`: Uses Yolo12x model to detect human in frames

    `ball`: Uses a custom trained Yolo11x model to detect the ball in frames. Training was done on Roboflow and is publicly available via  `model_id = pickleball-vision/6`

    `court`: Uses a custom trained Yolo11x model and Yolo11 model. Court keypoint is not robust at the moment.

3. Detection Results:

    Saves the detections onto a stub file for efficiency. When the video (with the same file name) is run again, it loads the stub file to save inference and processing time.

4. Filtering and Annotation

    Filters player detections to focus on the 4 human closest to the court, and draws bounding boxes and frame numbers on the output frames.

5. Video Output

    Saves the annotated frames as a new output video file.

### How to run this application?

To run this application, follow these steps:

1. Clone the Repository:

    Open your terminal or command prompt.
    Use git clone to download the project files.

    ```
    git clone https://github.com/d3smondwong/pickleball_video_player_ball_tracker.git
    ```
2. Navigate into the cloned project directory:

    ```
    cd [your_project_directory]
    ```
3. Set up Environment Variables:

    a. Create a `.env` file in the project directory

    b. Open the newly created .env file with a text editor and add your     ROBOFLOW API Key:

    ```
    ROBOFLOW_API_KEY="YOUR_API_KEY_HERE"
    ```
    &nbsp;&nbsp;&nbsp;&nbsp;Replace "YOUR_API_KEY_HERE" with your actual Roboflow API key.

4. Install Dependencies:

    a. Ensure you are still in the project's root directory (where requirements.txt is located).

    b. Run the following command to install all necessary Python libraries:

    ```
    pip install -r requirements.txt
    ```

5. Put the video you will like the application to detect the player and ball in the folder `input_videos`. Supported formats are `.mp4`, `.avi` and `.mov`

6. Run the application:

    From your project's root directory in the terminal, execute:

    ```
    python -m src.app

    ```
    This will run the application and output the video with both player and ball detection in the folder `output_videos`. The output file name will be the combination of input file name and output to give `{input file name}_output.avi`