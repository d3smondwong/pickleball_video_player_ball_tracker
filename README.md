# Pickleball Player and Ball Tracker

This computer vision application automates pickleball game analysis by tracking player movements and ball trajectories. It utilizes the YOLOv12x model for robust player detection and a custom, Roboflow-trained model for ball tracking. The system then employs perspective projection to map these dynamic, real-world positions onto a 2D court model, automatically calculating in-game statistics like ball hits and player running distance for interactive visualization.

### Use Case

Accurate tracking of players and the ball enables detailed analysis of individual performance and match dynamics. With reliable detection of court keypoints, the application can generate actionable insights and visualizations to help players improve their gameplay and strategy.

**Current Limitation:** Court keypoint detection is not yet robust. We will remove this feature for now.

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
├── analysis
|   └── ball_analysis.ipynb
├── data
│   ├── artifacts
│   │   ├── models
│   │   └── tracker_stubs
│   ├── court_keypoints
│   │   └── court_keypoints_{video_name}.json
│   ├── input_videos
│   └── output_videos
├── src
│   ├── court_line_detector
│   │   └── court_line_detector.py
│   ├── player_stats
│   │   └── player_stats.py
│   ├── mini_court
│   │   └── mini_court.py
│   ├── trackers
│   │   ├── __init__.py
│   │   ├── ball_tracker.py
│   │   └── player_tracker.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── conversions.py
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

2. Detection and tracking:

    `player`: Uses Yolo12x model to detect human in frames. It then uses ByteTrack to track players across the frames.

    `ball`: Uses a custom trained Yolo12x model to detect the ball in frames. Training was done on Roboflow and is publicly available via  `model_id = pickleball-vision/6`

    `court`: Uses a custom trained Yolo11x model and Yolo11 model. Court keypoint detection is not robust at the moment. More training data for finetuning is required.

3. Detection Results:

    Saves the detections onto a stub file for efficiency. When the video (with the same file name) is run again, it loads the stub file to save inference and processing time.

4. Mini-court

    `mini-court`: Draw a mini pickleball court on the frames

    `Coordinate conversion`: Use a homography matrix to map the ball to pixels on the mini-court from real-world perspective and court keypoints distance to calculate players position on the mini-court.

5. Player statistics

    Calculates the number of hits a player made and the distance the player move during the rally. Ball hits is tracked by checking a change in velocity, acceleration and distance moved. Distance is calculated by tracking the difference in foot position of the player across the frames.

6. Filtering and annotation

   `player`: Screen and filter the detections to focus on the 4 players. This is done using the foot positions which are closes to the court keypoints and within the defined court polygon calculated using Ray Casting algorithm. Custom annotations using an elipse at the players foot position and player's name are drawn for the players.

   `ball`: Custom annotation using a triangle to track the ball across frames

7. Saving detection results to a stub file

    To optimize development and avoid re-running expensive detection processes, detection results (e.g., bounding boxes, object IDs, and tracking data) are saved to a JSON stub file after initial processing of the video for the first time. During subsequent runs, the script can load from this stub file, skipping detection and allowing faster iteration on downstream features like team identification and annotation.

8. Video Output

    Saves the annotated frames as a new output video file.

9. Error Handling

    Throughout, the script checks for missing data and logs errors, ensuring robust execution.

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

6. Add in the court_keypoints in json format with the following naming convention

    ```
    court_keypoints_{video_name}.json

    ```

7. Run the application:

    From your project's root directory in the terminal, execute:

    ```
    python -m src.app

    ```
    This will run the application and output the video with both player and ball detection in the folder `output_videos`. The output file name will be the combination of input file name and output to give `{input file name}_output.avi`
