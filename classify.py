import os
import cv2
import numpy as np
from ultralytics import YOLO

def classify(model_path, input_path, output_path):
    # Validate paths
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to load image from: {input_path}")

    # Load YOLO model and class names
    model = YOLO(model_path)
    names = model.names

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Predict objects in the image
    results = model.predict(image, show=False)
    if results[0].boxes is None or len(results[0].boxes) == 0:
        print("No objects detected.")
        return

    # Extract bounding boxes and class indices
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()

    # Initialize counters for apple types
    red_apple, green_apple, yellow_apple = 0, 0, 0

    # Iterate through detections
    height, width = image.shape[:2]
    for box, cls in zip(boxes, clss):
        label = names[int(cls)]
        if label == "green apple":
            green_apple += 1
            jpg_filename = f"green_{green_apple}.jpg"
        elif label == "red apple":
            red_apple += 1
            jpg_filename = f"red_{red_apple}.jpg"
        elif label == "yellow apple":
            yellow_apple += 1
            jpg_filename = f"yellow_{yellow_apple}.jpg"
        else:
            continue

        # Ensure bounding box coordinates are within image bounds
        x_min = max(0, int(box[0]))
        y_min = max(0, int(box[1]))
        x_max = min(width, int(box[2]))
        y_max = min(height, int(box[3]))

        # Crop and save the object
        crop_obj = image[y_min:y_max, x_min:x_max]
        cv2.imwrite(os.path.join(output_path, jpg_filename), crop_obj)

if __name__ == "__main__":
    model_path = "model/yolo11n_best.pt"
    image_path = "input_classify/ai_assignment_20230726_classify.jpeg"
    output_path = "output_classify"
    classify(model_path=model_path, input_path=image_path, output_path=output_path)
