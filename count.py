import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from IPython.display import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
import cv2

def convert_bbox_format(bbox):
    """
    Converts bounding box format between from [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max].

    Args:
        bbox (list or tuple): The bounding box to convert. Format: [x_min, y_min, width, height] or [x_min, y_min, x_max, y_max]
    Returns:
        list: Converted bounding box in the target format.
    """
    # Convert from [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    return [x_min, y_min, x_max, y_max]

def get_box_class_from_predicted(predicted):
    """
    Extract boxes, classes, labels, and prediction accuracy from prediction result 
    Args:
        predicted: prediction result in sahi format
    Returns:
        boxs,classes,labels and scores. all variables is a list
    """
    output_detected = predicted.to_coco_annotations()
    boxes, classes, labels, scores = [],[],[],[]
    for i in range(len(output_detected)):
        boxes.append(convert_bbox_format(output_detected[i]["bbox"]))
        classes.append(output_detected[i]["category_id"])
        labels.append(output_detected[i]["category_name"])
        scores.append(output_detected[i]["score"])
    return boxes, classes, labels, scores

def predict(model_path, image_path, output_path):
    #read image
    image = cv2.imread(image_path)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.2,
        device="cuda:0",  # or 'cuda:0'
    )

    #do prediction using sliced prediction
    prediction_result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=740,
        slice_width=740,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,)
    
    # Access the object prediction list
    boxes, classes, labels, scores = get_box_class_from_predicted(prediction_result)
    names = YOLO(model_path).names

    # Prepare directory for output image
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Annote the image
    annotator = Annotator(image, line_width=2, example=names)
    counted = 0
    print(f"There are {len(boxes)} oil palm tree detected.")
    for i in range(len(boxes)):
        box, cls, acc = boxes[i], classes[i], scores[i]
        acc = np.round(acc, 3)
        label=names[int(cls)]
        #annoted only for oil palm tree
        if label == "Oil-Palm-Tree":
            counted+=1
            # Annotate the frame
            annotator.box_label(box, color=colors(int(cls), True), label=f"{counted}")
            #write annoted image as jpeg
    cv2.imwrite(os.path.join(output_path,"count.jpg"), image)

if __name__ == "__main__":
    model_path = "model/yolo11n_best.pt"
    image_path = "input_count/ai_assignment_20241202_count.jpeg"
    output_path = "output_count"
    predict(model_path=model_path, image_path=image_path, output_path=output_path)