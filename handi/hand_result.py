import dataclasses
import math
from typing import Iterable

import mediapipe.python.solutions.hands_connections as hc
from mediapipe.tasks.python.components.containers import NormalizedLandmark, Category
from mediapipe.tasks.python.vision import HandLandmarkerResult

FINGERS = [
    hc.HAND_THUMB_CONNECTIONS,
    hc.HAND_INDEX_FINGER_CONNECTIONS,
    hc.HAND_MIDDLE_FINGER_CONNECTIONS,
    hc.HAND_RING_FINGER_CONNECTIONS,
    hc.HAND_PINKY_FINGER_CONNECTIONS,
]

FINGER_NAMES = [
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
]


@dataclasses.dataclass
class HandResult:
    handedness: str
    finger_sizes: list[float]
    finger_angles: list[float]
    finger_centers: list[tuple[float, float]]
    size_sq: float
    center: tuple[float, float]

    @property
    def is_left(self):
        return self.handedness == "left"

    @property
    def size(self) -> float:
        return math.sqrt(self.size_sq)


def get_hand_results_from_landmarker(
    result: HandLandmarkerResult,
) -> Iterable[HandResult]:
    for handedness, landmarks in zip(result.handedness, result.hand_landmarks):
        yield process_single_hand(handedness, landmarks)


def process_single_hand(
    handedness_categories: list[Category],
    landmarks: list[NormalizedLandmark],
):
    handedness = handedness_categories[0].category_name.lower()
    x_coordinates = [landmark.x for landmark in landmarks]
    y_coordinates = [landmark.y for landmark in landmarks]
    x0, y0, x1, y1 = (
        min(x_coordinates),
        min(y_coordinates),
        max(x_coordinates),
        max(y_coordinates),
    )
    size_sq = (x1 - x0) ** 2 + (y1 - y0) ** 2
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    finger_sizes = []
    finger_angles = []
    finger_centers = []
    for index, connections in enumerate(FINGERS):
        length_sq = 0
        for va, vb in connections[1:]:  # we don't care about the first connection
            dxsq = (landmarks[vb].x - landmarks[va].x) ** 2
            dysq = (landmarks[vb].y - landmarks[va].y) ** 2
            length_sq += dxsq + dysq
        finger_sizes.append(round(math.sqrt(length_sq) * 1000, 1))
        base_vtx_idx = connections[1][0]
        tip_vtx_idx = connections[-1][1]
        base_vtx = landmarks[base_vtx_idx]
        tip_vtx = landmarks[tip_vtx_idx]
        finger_angles.append(
            math.degrees(math.atan2(tip_vtx.y - base_vtx.y, tip_vtx.x - base_vtx.x))
        )
        finger_centers.append(
            (
                base_vtx.x + (tip_vtx.x - base_vtx.x) / 2,
                base_vtx.y + (tip_vtx.y - base_vtx.y) / 2,
            )
        )
    return HandResult(
        handedness=handedness,
        finger_sizes=finger_sizes,
        finger_angles=finger_angles,
        finger_centers=finger_centers,
        size_sq=size_sq,
        center=center,
    )
