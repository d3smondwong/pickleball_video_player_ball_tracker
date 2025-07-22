import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 16)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        # Define the transformations to apply to the input image
        # Convert the image to numpy array, resize it (224, 224), convert it to tensor
        # Normalize it with pretrained ResNet50 model mean and std
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        # Convert the image to RGB as model expects RGB input
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply the transformations
        # Add a batch dimension to the image tensor
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Run inference by disabling gradient descent calculation to save memory and get the outputs
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Convert the outputs to keypoints, squeeze the batch dimension to remove any unnecessary dimensions
        # move to GPU if available
        if torch.cuda.is_available():
            outputs = outputs.cuda()

        # Squeeze the outputs to remove the batch dimension, and move to CPU to convert to numpy
        keypoints = outputs.squeeze().cpu().numpy()

        # Extract the original image dimensions
        original_h, original_w = image.shape[:2]

        # Rescale the keypoints to the original image dimensions
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames