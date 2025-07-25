import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
from torchvision.models import ResNet50_Weights
import numpy as np
from supervision.detection.core import Detections
from inference_sdk import InferenceHTTPClient

class CourtLineDetector:
    """
    CourtLineDetector is a class for detecting court lines and keypoints in images or video frames using either a custom PyTorch model or a Roboflow inference API.
    Depending on initialization, the class supports two modes:
    1. PyTorch Model Mode: Loads a pre-trained ResNet50-based model for keypoint regression, suitable for local inference.
    2. Roboflow API Mode: Uses Roboflow's InferenceHTTPClient to perform remote inference for keypoint detection.
    Methods:
        __init__(model_path) or __init__(api_key, model_id):
            Initializes the detector with either a local model or Roboflow API credentials.
        predict(image):
            Runs inference on the input image and returns detected keypoints as a list or numpy array.
        draw_keypoints(image, keypoints):
            Draws keypoints on the given image, supporting both Roboflow and YOLO keypoint formats.
        draw_keypoints_on_video(video_frames, keypoints):
            Draws keypoints on each frame of a video sequence.
    Attributes:
        device (torch.device): The device used for PyTorch inference (if applicable).
        model (torch.nn.Module): The loaded PyTorch model (if applicable).
        transform (torchvision.transforms.Compose): Image preprocessing pipeline (if applicable).
        client (InferenceHTTPClient): Roboflow inference client (if applicable).
        model_id (str): Roboflow model identifier (if applicable).
    """

    # CourtLineDetector class for detecting court lines in images using a Roboflow model.
    def __init__(self, api_key: str = None, model_id: str = None, model_path: str = None):
        if api_key is not None and model_id is not None:
            # Roboflow API mode
            self.client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key
            )
            self.model_id = model_id
            self.model = None
            self.transform = None
            self.device = None
        elif model_path is not None:
            # Local PyTorch model mode
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 12 * 2)  # Assuming 12 keypoints with x, y coordinates
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            self.model.to(self.device)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.client = None
            self.model_id = None
        else:
            raise ValueError("You must provide either (api_key and model_id) for Roboflow mode or model_path for local PyTorch mode.")

    def predict_local_model(self, image):

        if self.model is None or self.transform is None:
            raise RuntimeError("Local model mode is not initialized. Please provide a model_path when creating CourtLineDetector.")

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

        keypoints = keypoints.reshape(-1, 2)

        return keypoints

    def predict_roboflow(self, image: np.ndarray) -> list:
        """
        Predict court keypoints for a given image using the Roboflow model.

        Args:
            image (np.ndarray): The input image in BGR format.

        Returns:
            np.ndarray: An array of keypoints in the format [[x1, y1], [x2, y2], ...].
        """
        # Run inference on the image
        result = self.client.infer(inference_input=image, model_id=self.model_id)

        # Handle result as a list or dict depending on API response
        if isinstance(result, list) and len(result) > 0 and 'predictions' in result[0]:
            prediction = result[0]['predictions'][0] if len(result[0]['predictions']) > 0 else {}
        elif isinstance(result, dict) and 'predictions' in result and len(result['predictions']) > 0:
            prediction = result['predictions'][0]
        else:
            prediction = {}

        # Extract keypoints from predictions
        keypoints = prediction.get('keypoints', []) if prediction else []

        return keypoints

    # Method to draw keypoints on an image. Using both Roboflow and YOLO keypoints format
    def draw_keypoints(self, image, keypoints):
        for idx, kp in enumerate(keypoints):
            if isinstance(kp, dict):
                x = int(np.clip(kp['x'], 0, image.shape[1] - 1))
                y = int(np.clip(kp['y'], 0, image.shape[0] - 1))
            else:
                x = int(np.clip(kp[0], 0, image.shape[1] - 1))
                y = int(np.clip(kp[1], 0, image.shape[0] - 1))
            cv2.putText(image, str(idx), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames


