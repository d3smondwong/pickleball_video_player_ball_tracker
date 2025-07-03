import os
import torch

def train_model(data_yaml_path, epochs=100, imgsz=640, model='artifacts/models/yolo11x.pt'):
    """
    Trains a YOLO detection model.

    Args:
        data_yaml_path (str): Path to the data.yaml file.
        epochs (int): Number of training epochs.
        imgsz (int): Input image size.
        model (str): Pre-trained model to use (e.g., 'yolo11x.pt').
    """
    command = (
        f"yolo task=detect mode=train model={model} data={data_yaml_path} "
        f"epochs={epochs} imgsz={imgsz}"
    )
    print(f"Executing training command: {command}")
    os.system(command)

if __name__ == "__main__":

    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        training_device = '0' # Use the first GPU
    else:
        print("CUDA is NOT available. Training will default to CPU if no GPU is found.")
        training_device = 'cpu'

    # python -m training.model_training_ball
    dataset_location = 'training/Pickleball-Vision-1'
    data_yaml = os.path.join(dataset_location, 'data.yaml')

    training_epochs = 100
    image_size = 640
    yolo_model = 'yolo11x.pt'

    # Ensure the data.yaml path exists (optional, but good practice)
    if not os.path.exists(data_yaml):
        print(f"Error: data.yaml not found at {data_yaml}. Please check the path.")
    else:
        train_model(
            data_yaml_path=data_yaml,
            epochs=training_epochs,
            imgsz=image_size,
            model=yolo_model
        )