import numpy as np
from face_detector import YoloDetector
import cv2
import math
from arduino_control import ArduinoController


class FaceDetector:
    """비전 기반 얼굴 인식을 위한 클래스"""

    def __init__(self, model: YoloDetector):
        self.model: YoloDetector = model
        self.track: bool = True
        self.skip_flag: bool = False
        self.target_boundary: int = 70
        self.center_boundary: int = 90

    def run(
        self,
        frame: np.ndarray,
        controller: ArduinoController,
        mode: str = "bracket",
    ) -> None:
        """
        mode = bracket -> bracket control
        mode = rail -> rail control
        """

        # Get the frame dimensions
        height, width = frame.shape[:2]
        center_x = int(width / 2)
        center_y = int(height / 2)

        # Preprocess the frame for YOLOv5
        bboxes, points = self.model.predict(frame)
        for box in bboxes[0]:
            # Define parameters
            x, y, w, h = box
            target_center_x = int((x + w) / 2)
            target_center_y = int((y + h) / 2)
            # bracket mode
            if mode == "bracket":
                # Define distance and direction
                distance = math.sqrt(
                    (center_x - target_center_x) ** 2
                    + (center_y - target_center_y) ** 2
                )
                delta_x = round((center_x - target_center_x) / distance)
                delta_y = round((target_center_y - center_y) / distance)
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                # if object is in between the two circles and mode is 1
                if self.track and distance > self.target_boundary:
                    cv2.circle(
                        frame, (center_x, center_y), 10, (255, 0, 0), 1
                    )
                    cv2.circle(
                        frame,
                        (target_center_x, target_center_y),
                        self.target_boundary,
                        (0, 255, 0),
                        1,
                    )
                    if self.skip_flag:
                        self.skip_flag = False
                        continue
                    else:
                        controller.bracket(delta_x, delta_y)
                        self.skip_flag = True
                # if object gets closer to the inner circle, change mode
                elif self.track and distance <= self.target_boundary:
                    self.track = False
                # if object is in between the two circles and mode is 2
                if self.track is False and distance <= self.center_boundary:
                    cv2.circle(
                        frame, (center_x, center_y), 90, (255, 0, 0), 1
                    )
                    cv2.circle(
                        frame,
                        (target_center_x, target_center_y),
                        10,
                        (0, 255, 0),
                        1,
                    )
                # if object gets farther from the outer circle, change mode
                elif self.track is False and distance > self.target_boundary:
                    self.track = True
                print(
                    f"- [BRACKET] Distance: {distance}, Direction_x: {delta_x}, Direction_y: {delta_y}), track: {self.track}"  # noqa: E501
                )
            elif mode == "rail":
                distance = center_x - target_center_x
                velocity_percentage = round(abs(distance) / center_x * 100)
                controller.rail(distance, velocity_percentage)
                print(
                    f"- [RAIL] Distance: {distance}, Velocity: {velocity_percentage}%"
                )
            controller.read()


if __name__ == "__main__":
    detector = FaceDetector(
        model=YoloDetector(
            weights_name="yolov5n_state_dict.pt",
            config_name="yolov5n.yaml",
            target_size=480,
            device="cpu",
            min_face=10,
        )
    )
    # Open the video file
    cap = cv2.VideoCapture(4)  # Replace with the path to your video file
    controller = ArduinoController()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Face Detection in Video", frame)
        detector.run(frame=frame, controller=controller)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
            controller.reset()
            break

    cap.release()
    cv2.destroyAllWindows()
