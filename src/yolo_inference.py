from ultralytics import YOLO

model = YOLO('yolo11x')

result = model.predict('input_videos/frames_00304_jpg.rf.1bfef6c4799a2e6fbb850802256667b3.jpg', save=True)

# python -m src.yolo_inference

# input_videos\frames_00304_jpg.rf.1bfef6c4799a2e6fbb850802256667b3.jpg