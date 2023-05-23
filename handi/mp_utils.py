# Via https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
#     (license: Apache 2.0)

import cv2
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from mediapipe.python.solutions import hands, drawing_styles
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.tasks.python.vision import HandLandmarkerResult

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result: HandLandmarkerResult):
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for handedness, hand_landmarks in zip(
        detection_result.handedness, detection_result.hand_landmarks
    ):
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            hands.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image
