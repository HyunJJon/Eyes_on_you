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

from arduino_control import ArduinoController


class Hand(IntEnum):
    """손가락의 키포인트 인덱스를 정의한다."""

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
    """비전 기반 손 제스처 인식을 위한 클래스"""

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
        cam_id: int = 1,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        confidence_threshold: float = 0.7,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        # Load the gesture recognizer model
        self.model = load_model(model_name)  # type: ignore
        x, y, _ = self.xyc
        self.controller = ArduinoController(width=x, height=y)

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
        """사용 가능한 제스처 이름을 반환한다."""
        # Load class names
        with open("gesture.names", "r") as f:
            class_names = f.read().split("\n")
            # ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me',
            # 'stop', 'rock', 'live long', 'fist', 'smile']
        return class_names

    @property
    def frame(self) -> np.ndarray:
        """웹캠에서 1프레임을 읽어온다."""
        _, frame = self.cap.read()
        assert frame is not None, "No image found"
        return cv2.flip(frame, 1)

    @cached_property
    def xyc(self) -> Tuple[int, int, int]:
        """프레임의 크기를 반환한다. e.g. (width, height, channels)"""
        x, y, c = self.frame.shape
        return x, y, c

    def detect_hand_keypoints(
        self, stream: bool = False, show: bool = True
    ) -> None:
        """실시간으로 손과 손가락의 키포인트를 인식한다.
        그리고 인식된 제스처에 따라 콜백 함수를 호출한다."""
        x, y, _ = self.xyc
        hand_connections = list(mpHands.HAND_CONNECTIONS)

        # 제스처 이름과 콜백 함수를 매핑한 딕셔너리
        gesture_callbacks = {
            "peace": self.peace_callback,
            "okay": self.ok_callback,
            "thumbs up": self.thumbs_up_callback,
            "thumbs down": self.thumbs_down_callback,
            "stop": self.stop_callback,
        }

        # 메인 루프
        while True:
            # 프레임을 읽어온다.
            frame_bgr = self.frame

            # 손의 랜드마크를 인식한다.
            hand_landmarks = self.hands.process(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            ).multi_hand_landmarks  # type: ignore

            # 각 손에 대해 랜드마크를 인식한다. (손이 여러 개일 수 있다.)
            for hand_idx, hand_landmark in enumerate(hand_landmarks or []):
                # 랜드마크들의 좌표를 리스트로 변환한다. (x, y) 형태로 저장된다.
                landmarks = [
                    int(coord)
                    for lm in hand_landmark.landmark
                    for coord in (lm.x * x, lm.y * y)
                ]

                # 손의 랜드마크를 모델에 입력하여 제스처를 예측한다.
                prediction = self.model.predict(
                    np.array(landmarks).reshape(-1, 21, 2), verbose=0  # type: ignore  # noqa: E501
                )

                # 만약 예측된 제스처가 허용된 제스처 중 하나이면서 제스처의
                # 확률이 confidence_threshold보다 크면 제스처 이름을 얻는다.
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
                # 그렇지 않으면 빈 문자열을 얻는다.
                else:
                    class_name = ""

                # 만약 show가 True이면 손의 랜드마크와 제스처 이름을 화면에
                # 표시한다.
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

                # gesture_callbacks에 제스처 이름이 있으면 콜백 함수를 호출한다.
                if class_name in gesture_callbacks:
                    gesture_callbacks[class_name](frame_bgr, landmarks)

            # 루프의 마지막 부분
            if show:
                cv2.imshow("Output", frame_bgr)
            if stream:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                cv2.waitKey(0)
                break

        # 루프가 종료되면 웹캠을 해제한다.
        cv2.destroyAllWindows()
        self.cap.release()

    def get_x_y_of_finger(
        self, landmarks: List[int], finger_id: int
    ) -> Tuple[int, int]:
        """손가락의 x, y 좌표를 반환한다."""
        x, y, _ = self.xyc
        return (
            int(landmarks[finger_id * 2] * y / x),
            int(landmarks[finger_id * 2 + 1] * x / y),
        )

    def peace_callback(
        self, frame: np.ndarray, landmarks: List[int]
    ) -> None:
        """Peace 제스처가 인식되었을 때의 콜백 함수"""
        for finger_id in (Hand.INDEX_TIP, Hand.MIDDLE_TIP):
            finger_x, finger_y = self.get_x_y_of_finger(landmarks, finger_id)
            self.controller.send_x_y(finger_x, finger_y)
            cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 0), -1)

    def ok_callback(self, frame: np.ndarray, landmarks: List[int]) -> None:
        """Okay 제스처가 인식되었을 때의 콜백 함수"""
        for finger_id in (Hand.INDEX_TIP, Hand.THUMB_TIP):
            finger_x, finger_y = self.get_x_y_of_finger(landmarks, finger_id)
            cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 0), -1)

    def thumbs_up_callback(
        self, frame: np.ndarray, landmarks: List[int]
    ) -> None:
        """Thumbs up 제스처가 인식되었을 때의 콜백 함수"""
        for finger_id in (Hand.THUMB_TIP,):
            finger_x, finger_y = self.get_x_y_of_finger(landmarks, finger_id)
            cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 0), -1)

    def thumbs_down_callback(
        self, frame: np.ndarray, landmarks: List[int]
    ) -> None:
        """Thumbs down 제스처가 인식되었을 때의 콜백 함수"""
        for finger_id in (Hand.THUMB_TIP,):
            finger_x, finger_y = self.get_x_y_of_finger(landmarks, finger_id)
            cv2.circle(frame, (finger_x, finger_y), 10, (255, 0, 0), -1)
            self.controller.send_x_y(finger_x, finger_y)

    def stop_callback(self, frame: np.ndarray, landmarks: List[int]) -> None:
        """Stop 제스처가 인식되었을 때의 콜백 함수"""
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
    # 인식기 초기화
    detector = Detector(
        cam_id=0,
        confidence_threshold=0.3,  # 이 값보다 큰 제스처만 인식한다.
        max_num_hands=1,  # 인식할 손의 개수
        allowed_gestures={  # 인식할 제스쳐들. 이외의 제스쳐는 무시한다.
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

    # 인식 시작
    detector.detect_hand_keypoints(stream=True, show=True)
