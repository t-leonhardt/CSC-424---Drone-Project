import cv2
import os
import shutil
import yaml
from pathlib import Path

# 1. Create YOLOv4 Dataset Configuration Files
def create_yolo_config(dataset_path):
    obj_data = f"""classes = 2
    train = {dataset_path}/train.txt
    valid = {dataset_path}/val.txt
    names = {dataset_path}/obj.names
    backup = backup/
    """
    obj_names = "clean\ndirty\n"

    with open(Path(dataset_path) / 'obj.data', 'w') as f:
        f.write(obj_data)
    
    with open(Path(dataset_path) / 'obj.names', 'w') as f:
        f.write(obj_names)

    print("YOLOv4 dataset configuration files created.")

# 2. Generate file paths for training and validation
def generate_file_paths(dataset_path):
    train_images = list((Path(dataset_path) / 'train/images').glob("*.jpg"))
    val_images = list((Path(dataset_path) / 'val/images').glob("*.jpg"))

    with open(Path(dataset_path) / 'train.txt', 'w') as f:
        f.writelines([str(img) + "\n" for img in train_images])

    with open(Path(dataset_path) / 'val.txt', 'w') as f:
        f.writelines([str(img) + "\n" for img in val_images])

    print("Train and validation file lists created.")

# 3. Training YOLOv4
def train_model(dataset_path, cfg_path, weights_path):
    create_yolo_config(dataset_path)
    generate_file_paths(dataset_path)

    # Ensure darknet is installed and available
    os.system(f"./darknet detector train {dataset_path}/obj.data {cfg_path} {weights_path} -dont_show -map")

# 4. Real-time Inference with YOLOv4
def inference(model_cfg, model_weights, class_names):
    net = cv2.dnn.readNet(model_weights, model_cfg)

    # Open video stream (for drone camera)
    cap = cv2.VideoCapture(0)  # Change to drone camera

    with open(class_names, 'r') as f:
        classes = f.read().strip().split("\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        outputs = net.forward(layer_names)

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]

                if confidence > 0.5: 
                    center_x, center_y, width, height = (detection[:4] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).astype("int")
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, f"{classes[class_id]}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Solar Panel Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):       # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_path = ""  # Set your dataset directory
    cfg_path = "cfg/yolov4.cfg"  # Path to YOLOv4 config file
    weights_path = "yolov4.weights"  # Pretrained weights
    model_cfg = "cfg/yolov4.cfg"
    model_weights = "yolov4.weights"
    class_names = "obj.names"

    # Step 1: Train the model (comment out after first run)
    train_model(dataset_path, cfg_path, weights_path)

    # Step 2: Real-time Inference
    inference(model_cfg, model_weights, class_names)