import dataclasses
import math
from typing import Iterable

import mediapipe.python.solutions.hands_connections as hc
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.tasks.python.vision import HandLandmarkerResult

FINGERS = [
    hc.HAND_THUMB_CONNECTIONS,
    hc.HAND_INDEX_FINGER_CONNECTIONS,
    hc.HAND_MIDDLE_FINGER_CONNECTIONS,
    hc.HAND_RING_FINGER_CONNECTIONS,
    hc.HAND_PINKY_FINGER_CONNECTIONS,
]


@dataclasses.dataclass
class HandResult:
    finger_sizes: list[float]
    finger_angles: list[float]
    finger_centers: list[tuple[float, float]]
    size_sq: float
    center: tuple[float, float]


def get_hand_results_from_landmarker(
    result: HandLandmarkerResult,
) -> Iterable[HandResult]:
    for hand in result.hand_landmarks:
        yield process_single_hand(hand)


def process_single_hand(hand: list[NormalizedLandmark]):
    x_coordinates = [landmark.x for landmark in hand]
    y_coordinates = [landmark.y for landmark in hand]
    bbox = (
        min(x_coordinates),
        min(y_coordinates),
        max(x_coordinates),
        max(y_coordinates),
    )
    size_sq = (bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    finger_sizes = []
    finger_angles = []
    finger_centers = []
    for index, connections in enumerate(FINGERS):
        length_sq = 0
        for va, vb in connections[1:]:  # we don't care about the first connection
            dxsq = (hand[vb].x - hand[va].x) ** 2
            dysq = (hand[vb].y - hand[va].y) ** 2
            length_sq += dxsq + dysq
        finger_sizes.append(round(math.sqrt(length_sq) * 1000, 1))
        base_vtx_idx = connections[1][0]
        tip_vtx_idx = connections[-1][1]
        base_vtx = hand[base_vtx_idx]
        tip_vtx = hand[tip_vtx_idx]
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
        finger_sizes=finger_sizes,
        finger_angles=finger_angles,
        finger_centers=finger_centers,
        size_sq=size_sq,
        center=center,
    )
