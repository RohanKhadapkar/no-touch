import math
from array import array
from typing import Tuple

import cv2
import mediapipe as mp


class HandDetector:
    """
    This class is performs the hand detection.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Constructor function.

        Args:
            static_image_mode (bool, optional): Whether to treat the input images as batch of
            static images or a video stream. Defaults to False.
            max_num_hands (int, optional): Max number of hands allowed. Defaults to 1.
            min_detection_confidence (float, optional): Confidence value for the detection to be
            considered successful. Defaults to 0.5.
            min_tracking_confidence (float, optional): Confidence value for the detection to be
            considered successfu. Defaults to 0.5.
        """
        self.mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands_obj = self.mp_hands.Hands(
            self.mode,
            self.max_num_hands,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.finger_tip_ids = [4, 8, 12, 16, 20]
        self.hand_detection_results = []

    def find_hand(self, image: array, draw_landmarks: bool = True) -> array:
        """
        This function will find hands in the provided image and draw landmark (dots, dashes) on the
        fingers of the detected hand.

        Args:
            image (array): Webcam image, numpy array.
            draw_landmarks (bool, optional): Whether to draw landmarks or not. Defaults to True.

        Returns:
            array: Image with the drawn landmarks.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.hand_detection_results = self.hands_obj.process(rgb_image)

        if self.hand_detection_results.multi_hand_landmarks:
            for hand_landmarks in self.hand_detection_results.multi_hand_landmarks:
                if draw_landmarks:
                    self.mp_draw.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
        return image

    def find_hand_position(
        self, image: array, hand_no: int = 0, draw_landmarks: bool = True
    ) -> list:
        """
        This function will find the position of landmarks and return the postition along witht the
        bounding box.

        Args:
            image (array): Webcam image, numpy array.
            hand_no (int, optional): ID of the hand to be detected. Defaults to 0.
            draw_landmarks (bool, optional): Whether to draw landmarks or not. Defaults to True.

        Returns:
            Tuple[list, list]: Landmark list & bounding box.
        """
        x_axis_list = []
        y_axis_list = []
        self.landmark_list = []

        if self.hand_detection_results.multi_hand_landmarks:
            detected_hand = self.hand_detection_results.multi_hand_landmarks[hand_no]
            for landmark_id, landmark in enumerate(detected_hand.landmark):
                img_height, img_width, _ = image.shape
                x_axis_center, y_axis_center = int(landmark.x * img_width), int(
                    landmark.y * img_height
                )

                x_axis_list.append(x_axis_center)
                y_axis_list.append(y_axis_center)
                self.landmark_list.append([landmark_id, x_axis_center, y_axis_center])

                if draw_landmarks:
                    cv2.circle(image, (x_axis_center, y_axis_center), 7, (255, 255, 0), cv2.FILLED)

            x_min, x_max = min(x_axis_list), max(x_axis_list)
            y_min, y_max = min(y_axis_list), max(y_axis_list)

            if draw_landmarks:
                cv2.rectangle(
                    image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2
                )

        return self.landmark_list

    def find_raised_fingers(self) -> list:
        """
        This function will find which finger if any is raised for clicking.

        Returns:
            list: List of raised fingers.
        """
        raised_fingers = []
        if (
            self.landmark_list[self.finger_tip_ids[0]][1]
            > self.landmark_list[self.finger_tip_ids[0] - 1][1]
        ):
            raised_fingers.append(1)
        else:
            raised_fingers.append(0)

        for finger_id in range(1, 5):
            if (
                self.landmark_list[self.finger_tip_ids[finger_id]][2]
                < self.landmark_list[self.finger_tip_ids[finger_id] - 2][2]
            ):
                raised_fingers.append(1)
            else:
                raised_fingers.append(0)

        return raised_fingers

    def find_distance_between_fingers(
        self,
        index_finger_id: int,
        middle_finger_id: int,
        image: array,
        draw_landmarks: bool = True,
        landmark_circle_radius: int = 15,
        distance_line_thickness: int = 3,
    ) -> Tuple[int, array]:
        """
        This function will find the distance between index and middle finger to trigger event of
        clicking mouse button.

        Args:
            index_finger_id (int): ID of the index finger.
            middle_finger_id (int): ID of the middle finger.
            image (array): Webcam image, numpy array.
            draw_landmarks (bool, optional): Whether to draw landmarks or not. Defaults to True.
            landmark_circle_radius (int, optional): Radius of the landmark circle. Defaults to 15.
            distance_line_thickness (int, optional): Thickness of the line between fingers which
            depicts the distance. Defaults to 3.

        Returns:
            Tuple[int, array]: Distance between two fingers, image as numpy array.
        """
        index_finger_x_axis, index_finger_y_axis = self.landmark_list[index_finger_id][1:]
        middle_finger_x_axis, middle_finger_y_axis = self.landmark_list[middle_finger_id][1:]

        if draw_landmarks:
            cv2.line(
                image,
                (index_finger_x_axis, index_finger_y_axis),
                (middle_finger_x_axis, middle_finger_y_axis),
                (255, 0, 255),
                distance_line_thickness,
            )
            cv2.circle(
                image,
                (index_finger_x_axis, index_finger_y_axis),
                landmark_circle_radius,
                (255, 0, 255),
                cv2.FILLED,
            )
            cv2.circle(
                image,
                (middle_finger_x_axis, middle_finger_y_axis),
                landmark_circle_radius,
                (255, 0, 255),
                cv2.FILLED,
            )
            cv2.circle(
                image,
                (index_finger_x_axis, middle_finger_y_axis),
                landmark_circle_radius,
                (255, 0, 255),
                cv2.FILLED,
            )

        distance = math.hypot(
            middle_finger_x_axis, -index_finger_x_axis, middle_finger_y_axis - index_finger_y_axis
        )
        return (distance, image)
