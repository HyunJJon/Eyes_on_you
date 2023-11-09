from face_detector import YoloDetector
import numpy as np
from PIL import Image
import cv2
import math
import numpy as np
import time
from arduino_control import ArduinoController

# Open the video file
cap = cv2.VideoCapture(4)  # Replace with the path to your video file
model = YoloDetector(weights_name="yolov5n_state_dict.pt",config_name="yolov5n.yaml",target_size=480, device="cpu", min_face=10)
mode = 1
controller = ArduinoController()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Get the frame dimensions
    height, width = frame.shape[:2]

    # Preprocess the frame for YOLOv5
    img = frame  # No need to convert to RGB
    bboxes,points = model.predict(img)
    for box in bboxes[0]:
        x, y, w, h = box
        center_x = int(width/2)
        center_y = int(height/2)
        target_center_x = int((x+w)/2)
        target_center_y = int((y+h)/2)
        distance = math.sqrt((center_x-target_center_x)**2 + (center_y-target_center_y)**2)
        direction_x = -1 * round((target_center_x-center_x)/distance)
        direction_y = round((target_center_y-center_y)/distance)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        if mode == 1 and distance > 70:
            cv2.circle(frame, (center_x, center_y), 10, (255, 0, 0), 1)
            cv2.circle(frame, (target_center_x, target_center_y), 70, (0, 255, 0), 1)
            controller.send_x_y(direction_x, direction_y, mode)
        elif mode == 1 and distance <= 50:
            mode = 2
        if mode == 2 and distance <= 90:
            cv2.circle(frame, (center_x, center_y), 90, (255, 0, 0), 1)
            cv2.circle(frame, (target_center_x, target_center_y), 10, (0, 255, 0), 1)
        elif mode == 2 and distance > 70:
            mode = 1
        #time.sleep(0.1)
        print(f"Distance: {distance}, Direction_x: {direction_x}, Direction_y: {direction_y}), mode: {mode}")
    # print(f"Bounding Box Position: Top-Left ({x}, {y}), Bottom-Right ({x + w}, {y + h})")
    #controller.send_x_y(x+w/2, y+h/2)


    # Display the video with detected faces
    cv2.imshow("Face Detection in Video", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        controller.stop()
        break

# Release the video and close the OpenCV window
cap.release()
cv2.destroyAllWindows()