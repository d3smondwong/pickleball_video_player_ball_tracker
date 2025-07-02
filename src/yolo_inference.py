from ultralytics import YOLO

model = YOLO('artifacts/models/yolo12x.pt')

result = model.predict('input_videos/frames_00304_jpg.rf.1bfef6c4799a2e6fbb850802256667b3.jpg', save=True)

# To see the results. Understand the the structure of the result object.
# print(result)
# print("Boxes:")
# for box in result[0].boxes:
#     print(box)


# python -m src.yolo_inference

# input_videos\frames_00304_jpg.rf.1bfef6c4799a2e6fbb850802256667b3.jpg