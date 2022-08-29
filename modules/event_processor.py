import time

import autopy
import cv2
import numpy as np

from .constants import (
    CAM_HEIGHT,
    CAM_WIDTH,
    CURSOR_SMOOTHENING,
    INDEX_FINGER_ID,
    MIDDLE_FINGER_ID,
    RECTANGLE_FRAME,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_NAME,
    WINDOW_WIDTH,
)
from .hand_tracking import HandDetector


def event_handler() -> None:
    """
    This function will start the app process.
    """
    previous_time, current_time = 0, 0
    previous_location_x, previous_location_y = 0, 0
    current_location_x, current_location_y = 0, 0

    video_capture = cv2.VideoCapture(1)
    video_capture.set(3, CAM_WIDTH)
    video_capture.set(4, CAM_HEIGHT)

    hand_detector = HandDetector(max_num_hands=1)

    while True:
        _, captured_image = video_capture.read()
        image = hand_detector.find_hand(captured_image)
        landmark_list = hand_detector.find_hand_position(image)

        if len(landmark_list) != 0:
            cv2.rectangle(
                image,
                (RECTANGLE_FRAME, RECTANGLE_FRAME),
                (CAM_WIDTH - RECTANGLE_FRAME, CAM_HEIGHT - RECTANGLE_FRAME),
                (255, 255, 0),
                2,
            )

            index_finger_x_axis, index_finger_y_axis = landmark_list[INDEX_FINGER_ID][1:]
            middle_finger_x_axis, middle_finger_y_axis = landmark_list[MIDDLE_FINGER_ID][1:]
            raised_fingers = hand_detector.find_raised_fingers()

            if raised_fingers[1] == 1 and raised_fingers[2] == 0:
                center_x_axis = np.interp(
                    index_finger_x_axis,
                    (RECTANGLE_FRAME, CAM_WIDTH - RECTANGLE_FRAME),
                    (0, SCREEN_WIDTH),
                )
                center_y_axis = np.interp(
                    index_finger_y_axis,
                    (RECTANGLE_FRAME, CAM_HEIGHT - RECTANGLE_FRAME),
                    (0, SCREEN_HEIGHT),
                )

                center_loc_x = (
                    previous_location_x + (center_x_axis - previous_location_x) / CURSOR_SMOOTHENING
                )
                center_loc_y = (
                    previous_location_y + (center_y_axis - previous_location_y) / CURSOR_SMOOTHENING
                )
                autopy.mouse.move(SCREEN_WIDTH - center_loc_x, center_loc_y)
                cv2.circle(
                    image, (index_finger_x_axis, index_finger_y_axis), 15, (255, 255, 0), cv2.FILLED
                )
                previous_location_x, previous_location_y = center_loc_x, center_loc_y

            if raised_fingers[1] == 1 and raised_fingers[2] == 1:
                distance, image = hand_detector.find_distance_between_fingers(
                    INDEX_FINGER_ID, MIDDLE_FINGER_ID, image
                )
                print("Distance between two fingers: ", distance)

                if distance < 800:
                    cv2.circle(
                        image,
                        (index_finger_x_axis, index_finger_y_axis),
                        15,
                        (0, 255, 0),
                        cv2.FILLED,
                    )
                    cv2.circle(
                        image,
                        (middle_finger_x_axis, middle_finger_y_axis),
                        15,
                        (0, 255, 0),
                        cv2.FILLED,
                    )
                    autopy.mouse.click()

        # Draw FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(
            image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 255, 0), 3
        )
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(WINDOW_NAME, image)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.waitKey(1)
