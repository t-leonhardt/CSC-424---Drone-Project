import torch
import cv2
import os
import shutil
import yaml
from pathlib import Path

# 1. Dataset Preparation
# Ensure the images are organized in a directory structure like:
# /dataset/train/clean/ and /dataset/train/dirty/
# /dataset/val/clean/ and /dataset/val/dirty/

# 2. Create YOLO Dataset YAML configuration
def create_yaml(dataset_path):
    data = {
        'train': str(Path(dataset_path) / 'train'),
        'val': str(Path(dataset_path) / 'val'),
        'nc': 2,  # Number of classes (clean and dirty)
        'names': ['clean', 'dirty']
    }
    yaml_path = Path(dataset_path) / 'solar_panel_data.yaml'
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

# 3. Install Dependencies
# Install YOLOv5 dependencies if not already done
# pip install torch torchvision torchaudio
# pip install yolov5

# 4. Training Code
def train_model(dataset_path):
    # Prepare the dataset YAML file
    create_yaml(dataset_path)
    print("Dataset YAML created.")

    # 4.1. Training with YOLOv5
    os.system(f"python train.py --img 640 --batch 16 --epochs 50 --data {dataset_path}/solar_panel_data.yaml --weights yolov5n.pt --cache")

    # After training, the best model will be saved in runs/train/exp/weights/best.pt

# 5. Real-time Inference (for Drone)
def inference(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # Load the best model

    # Open video stream (for drone camera)
    cap = cv2.VideoCapture(0)  # needs to be changed to actual drone camera 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)
        
        # Render results on the frame
        frame = results.render()[0]  # Render bounding boxes and labels on the frame

        # Display the results
        cv2.imshow("Solar Panel Classification", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # 6. Paths
    dataset_path = ""  
    model_path = "runs/train/exp/weights/best.pt"  # Path to your trained model (after training)

    # Step 1: Train the model (comment this out after first run)
    train_model(dataset_path)

    # Step 2: Real-time Inference
    inference(model_path)