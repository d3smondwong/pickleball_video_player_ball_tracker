from ultralytics import YOLO

model = YOLO('artifacts/models/yolo12x.pt')

# To perform inference on a single image, use the `predict` method.
# result = model.predict('input_videos/frames_00304_jpg.rf.1bfef6c4799a2e6fbb850802256667b3.jpg', save=True)

# To perform tracking on a video or a sequence of images, use the `track` method.
result = model.track('input_videos/input_video.mp4', save=True)

# To see the results. Understand the the structure of the result object.
# print(result)
# print("Boxes:")
# for box in result[0].boxes:
#     print(box)


# python -m src.yolo_inference

# input_videos\frames_00304_jpg.rf.1bfef6c4799a2e6fbb850802256667b3.jpg