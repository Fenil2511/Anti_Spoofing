import math
import time
import cv2
import cvzone
from ultralytics import YOLO

# Initialize video capture
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)

# Load the model
yolo_model = YOLO({#Write path to your train model#})

# Define class names
class_names = ["fake", "real"]

# Initialize frame time variables
previous_frame_time = 0
current_frame_time = 0

while True:
    current_frame_time = time.time()
    read_success, image = video_capture.read()
    detection_results = yolo_model(image, stream=True, verbose=False)
    for result in detection_results:
        bounding_boxes = result.boxes
        for bounding_box in bounding_boxes:
            # Bounding Box
            x_start, y_start, x_end, y_end = bounding_box.xyxy[0]
            x_start, y_start, x_end, y_end = int(x_start), int(y_start), int(x_end), int(y_end)
            width, height = x_end - x_start, y_end - y_start
            # Confidence
            confidence_score = math.ceil((bounding_box.conf[0] * 100)) / 100
            # Class Name
            class_index = int(bounding_box.cls[0])
            if confidence_score > confidence:
                if class_names[class_index] == 'real':
                    box_color = (0, 255, 0)
                else:
                    box_color = (0, 0, 255)
                cvzone.cornerRect(image, (x_start, y_start, width, height), colorC=box_color, colorR=box_color)
                cvzone.putTextRect(image, f'{class_names[class_index].upper()} {int(confidence_score*100)}%', (max(0, x_start), max(35, y_start)), scale=2, thickness=4, colorR=box_color, colorB=box_color)

    frames_per_second = 1 / (current_frame_time - previous_frame_time)
    previous_frame_time = current_frame_time
    print(frames_per_second)

    cv2.imshow("Output", image)
    cv2.waitKey(1)
