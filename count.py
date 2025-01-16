import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
import cv2
#import torch

def convert_bbox_format(bbox):
    """Converts [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]."""
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    return [x_min, y_min, x_max, y_max]

def get_box_class_from_predicted(predicted):
    """Extract boxes, classes, labels, and scores from prediction result."""
    output_detected = predicted.to_coco_annotations()
    boxes, classes, labels, scores = [], [], [], []
    for i in range(len(output_detected)):
        boxes.append(convert_bbox_format(output_detected[i]["bbox"]))
        classes.append(output_detected[i]["category_id"])
        labels.append(output_detected[i]["category_name"])
        scores.append(output_detected[i]["score"])
    return boxes, classes, labels, scores

def predict(model_path, image_path, output_path):
    # Ensure files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Set device
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Load detection model
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=0.2,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load detection model: {e}")

    # Perform sliced prediction
    prediction_result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=740,
        slice_width=740,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    if not prediction_result:
        raise RuntimeError("Prediction result is empty.")

    # Extract predictions
    boxes, classes, labels, scores = get_box_class_from_predicted(prediction_result)
    try:
        names = YOLO(model_path).names
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")

    # Prepare output directory
    os.makedirs(output_path, exist_ok=True)

    # Annotate image
    annotator = Annotator(image, line_width=2, example=names)
    counted = 0
    print(f"There are {len(boxes)} oil palm trees detected.")
    for i in range(len(boxes)):
        box, cls, acc = boxes[i], classes[i], scores[i]
        label = names[int(cls)]
        if label == "Oil-Palm-Tree":
            counted += 1
            annotator.box_label(box, color=colors(int(cls), True), label=f"{counted}")

    # Save annotated image
    cv2.imwrite(os.path.join(output_path, "count.jpg"), annotator.result())

if __name__ == "__main__":
    model_path = "model/yolo11n_best.pt"
    image_path = "input_count/ai_assignment_20241202_count.jpeg"
    output_path = "output_count"
    predict(model_path=model_path, image_path=image_path, output_path=output_path)
