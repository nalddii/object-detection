import os
import cv2
import numpy as np
from ultralytics import YOLO

def classify(model_path, input_path, output_path):
    # Load YOLO model and class names
    model = YOLO(model_path)
    names = model.names

    # Prepare directory for cropped objects
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    image = cv2.imread(input_path)

    # Predict objects on the first frame
    results = model.predict(image, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()

    #define total detected for each color
    red_apple, green_apple, yellow_apple = 0,0,0

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            label=names[int(cls)]
            if label == "green apple":
                green_apple+=1
                jpg_filename = f"green_{green_apple}.jpg"
            elif label == "red apple":
                red_apple+=1
                jpg_filename = f"red_{red_apple}.jpg"
            elif label == "yellow apple":
                yellow_apple+=1
                jpg_filename = f"yellow_{yellow_apple}.jpg"
            else:
                continue

            # Crop and save the object
            crop_obj = image[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
            cv2.imwrite(os.path.join(output_path,jpg_filename), crop_obj)

if __name__ == "__main__":
    model_path = "model/yolo11n_best.pt"
    image_path = "input_classify/ai_assignment_20230726_classify.jpeg"
    output_path = "output_classify"
    classify(model_path=model_path, input_path=image_path, output_path=output_path)

    
