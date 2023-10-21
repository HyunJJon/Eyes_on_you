"""hand gesture recognition project using Python OpenCV"""

from enum import IntEnum
from functools import cached_property
from typing import List, Tuple

import cv2
import keras
import numpy as np
from keras.models import load_model
from mediapipe.python.solutions import drawing_utils as mpDraw
from mediapipe.python.solutions import hands as mpHands


class Hand(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class Detector:
    model: keras.Model
    hands: mpHands.Hands
    cap: cv2.VideoCapture

    def __init__(
        self,
        model_name: str = "mp_hand_gesture",
        allowed_gestures: set[str] = {
            "stop",
            "thumbs up",
            "thumbs down",
            "peace",
            "call me",
        },
        cam_id: int = 0,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        confidence_threshold: float = 0.7,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        # Load the gesture recognizer model
        self.model = load_model(model_name)  # type: ignore

        # initialize mediapipe hands for hand gesture recognition project
        self.hands = mpHands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Initialize the webcam for Hand Gesture Recognition Python project
        self.cap = cv2.VideoCapture(cam_id)
        assert self.cap.isOpened(), "Cannot open webcam"

        # Convert allowed gestures to their corresponding indices
        self.allowed_indices = np.array(
            [
                self.gesture_names.index(gesture)
                for gesture in allowed_gestures
                if gesture in self.gesture_names
            ]
        )

        self.confidence_threshold = confidence_threshold

    @cached_property
    def gesture_names(self) -> List[str]:
        # Load class names
        with open("gesture.names", "r") as f:
            class_names = f.read().split("\n")
            # ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me',
            # 'stop', 'rock', 'live long', 'fist', 'smile']
        return class_names

    @property
    def frame(self) -> np.ndarray:
        _, frame = self.cap.read()
        assert frame is not None, "No image found"
        return cv2.flip(frame, 1)

    @cached_property
    def xyc(self) -> Tuple[int, int, int]:
        x, y, c = self.frame.shape
        return x, y, c

    def detect_hand_keypoints(
        self, stream: bool = False, show: bool = True
    ) -> None:
        x, y, _ = self.xyc
        hand_connections = list(mpHands.HAND_CONNECTIONS)

        # Dictionary for gesture callbacks
        gesture_callbacks = {
            "peace": self.peace_callback,
            "okay": self.ok_callback,
            "thumbs up": self.thumbs_up_callback,
            "thumbs down": self.thumbs_down_callback,
            "stop": self.stop_callback,
        }

        while True:
            frame_bgr = self.frame
            hand_landmarks = self.hands.process(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            ).multi_hand_landmarks  # type: ignore

            for hand_idx, hand_landmark in enumerate(hand_landmarks or []):
                landmarks = [
                    int(coord)
                    for lm in hand_landmark.landmark
                    for coord in (lm.x * x, lm.y * y)
                ]
                prediction = self.model.predict(
                    np.array(landmarks).reshape(-1, 21, 2), verbose=0  # type: ignore  # noqa: E501
                )
                if (
                    np.max(prediction[hand_idx][self.allowed_indices])
                    >= self.confidence_threshold
                ):
                    # Get the class id using argmax
                    class_name = self.gesture_names[
                        self.allowed_indices[
                            np.argmax(
                                prediction[hand_idx][self.allowed_indices]
                            )
                        ]
                    ]
                else:
                    class_name = ""

                if show:
                    mpDraw.draw_landmarks(
                        frame_bgr, hand_landmark, hand_connections
                    )
                    cv2.putText(
                        frame_bgr,
                        class_name,
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # Use the callback from the dictionary
                if class_name in gesture_callbacks:
                    gesture_callbacks[class_name](frame_bgr, landmarks)

            if show:
                cv2.imshow("Output", frame_bgr)
            if stream:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                cv2.waitKey(0)
                break

        cv2.destroyAllWindows()
        self.cap.release()

    def get_x_y_of_finger(
        self, landmarks: List[int], finger_id: int
    ) -> Tuple[int, int]:
        x, y, _ = self.xyc
        return (
            int(landmarks[finger_id * 2] * y / x),
            int(landmarks[finger_id * 2 + 1] * x / y),
        )

    def peace_callback(
        self, frame: np.ndarray, landmarks: List[int]
    ) -> None:
        for finger_id in (Hand.INDEX_TIP, Hand.MIDDLE_TIP):
            finger_x, finger_y = self.get_x_y_of_finger(landmarks, finger_id)
            cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 0), -1)

    def ok_callback(self, frame: np.ndarray, landmarks: List[int]) -> None:
        for finger_id in (Hand.INDEX_TIP, Hand.THUMB_TIP):
            finger_x, finger_y = self.get_x_y_of_finger(landmarks, finger_id)
            cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 0), -1)

    def thumbs_up_callback(
        self, frame: np.ndarray, landmarks: List[int]
    ) -> None:
        for finger_id in (Hand.THUMB_TIP,):
            finger_x, finger_y = self.get_x_y_of_finger(landmarks, finger_id)
            cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 0), -1)

    def thumbs_down_callback(
        self, frame: np.ndarray, landmarks: List[int]
    ) -> None:
        for finger_id in (Hand.THUMB_TIP,):
            finger_x, finger_y = self.get_x_y_of_finger(landmarks, finger_id)
            cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 0), -1)

    def stop_callback(self, frame: np.ndarray, landmarks: List[int]) -> None:
        for finger_id in (
            Hand.THUMB_TIP,
            Hand.INDEX_TIP,
            Hand.MIDDLE_TIP,
            Hand.RING_TIP,
            Hand.PINKY_TIP,
        ):
            finger_x, finger_y = self.get_x_y_of_finger(landmarks, finger_id)
            cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 0), -1)


if __name__ == "__main__":
    detector = Detector(
        confidence_threshold=0.3,
        max_num_hands=1,
        allowed_gestures={
            "okay",
            "peace",
            "thumbs up",
            "thumbs down",
            # "call me",
            "stop",
            # "rock",
            # "live long",
            # "fist",
            # "smile",
        },
    )
    detector.detect_hand_keypoints(stream=True, show=False)

    # def get_one_frame(self) -> np.ndarray:
    #     cap = self.cap
    #     try:
    #         while True:
    #             # Read each frame from the webcam
    #             _, frame = cap.read()
    #             self.width, self.height, self.channels = frame.shape
    #             assert frame is not None, "No image found"

    #             # Flip the frame vertically
    #             frame = cv2.flip(frame, 1)
    #             # Show the final output
    #             cv2.imshow("Output", frame)
    #             if cv2.waitKey(1) == ord("q"):
    #                 break
    #     finally:
    #         # release the webcam and destroy all active windows
    #         cap.release()
    #         cv2.destroyAllWindows()
    #     return frame
