import os.path
import time

import mido
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode,
)
import cv2

from handi.controller_values import compute_ctl_values, send_changed_values
from handi.cv_utils import get_cam_frame
from handi.mp_utils import draw_landmarks_on_image
from handi.hand_result import get_hand_results_from_landmarker

# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
HAND_LANDMARKER_TASK = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def open_output():
    mido.set_backend("mido.backends.portmidi")
    outputs = mido.get_output_names()
    output = outputs[0]
    print(f"Using output {output}")
    return mido.open_output(output)


class HandiApp:
    def __init__(self):
        self.last_landmarker_result = None
        self.last_hands = []
        self.last_ctl_values = {}
        self.last_timestamp = None
        self.output_port = None

    def _process_result(
        self,
        result: HandLandmarkerResult,
        output_image: Image,
        timestamp_ms: int,
    ):
        self.last_landmarker_result = result
        self.last_hands = list(get_hand_results_from_landmarker(result))
        self.last_timestamp = timestamp_ms

    def get_landmarker(self):
        return HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                num_hands=2,
                base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_TASK),
                running_mode=RunningMode.LIVE_STREAM,
                result_callback=self._process_result,
            )
        )

    def run(self):
        self.output_port = open_output()

        cam = cv2.VideoCapture(0)

        with self.get_landmarker() as landmarker:
            while True:
                timestamp = int(time.time() * 1000)
                frame = get_cam_frame(cam)
                mp_image = Image(image_format=ImageFormat.SRGB, data=frame)
                landmarker.detect_async(mp_image, timestamp)
                if self.last_landmarker_result is not None:
                    # Could be slightly delayed but whatever
                    frame = draw_landmarks_on_image(
                        mp_image.numpy_view(),
                        self.last_landmarker_result,
                    )
                cv2.imshow("Frame", frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                if self.last_hands:
                    new_ctl_values = {}
                    for hand in self.last_hands:
                        new_ctl_values.update(compute_ctl_values(hand))
                    send_changed_values(
                        self.output_port, self.last_ctl_values, new_ctl_values
                    )
                    self.last_ctl_values = new_ctl_values


def main():
    app = HandiApp()
    app.run()


if __name__ == "__main__":
    main()
