import mido

from handi.hand_result import HandResult

BASE_CTL_NUM = 18
NUM_POS_VALUES = 10
NUM_ANG_VALUES = 5
NUM_SIZE_VALUES = 5
CTLS_PER_HAND = NUM_POS_VALUES + NUM_ANG_VALUES + NUM_SIZE_VALUES
MAX_CTL_NUM = BASE_CTL_NUM + CTLS_PER_HAND * 2
CTL_NUM_RANGE = range(BASE_CTL_NUM, MAX_CTL_NUM + 1)


def compute_ctl_values(hand: HandResult):
    base_ctl = BASE_CTL_NUM
    if hand.is_left:
        base_ctl += CTLS_PER_HAND
    pos_base_ctl = base_ctl
    ang_base_ctl = base_ctl + NUM_POS_VALUES
    size_base_ctl = base_ctl + NUM_POS_VALUES + NUM_ANG_VALUES
    ctl_values = {}
    for i, center in enumerate(hand.finger_centers):
        x, y = center
        x_value = max(0, min(127, round(x * 127)))
        y_value = max(0, min(127, round(y * 127)))
        ctl_values[pos_base_ctl + i * 2] = x_value
        ctl_values[pos_base_ctl + 1 + i * 2] = y_value
    for i, angle in enumerate(hand.finger_angles):
        angle_value = max(0, min(127, round((angle + 90) / 120 * 127)))
        ctl_values[ang_base_ctl + i] = angle_value
    for i, size in enumerate(hand.finger_sizes):
        size_value = max(0, min(127, round(size / 100 * 127)))
        ctl_values[size_base_ctl + i] = size_value
    return ctl_values


def send_changed_values(port, last_values, new_values):
    diff_ctl_values = {
        ctl: value
        for ctl, value in new_values.items()
        if ctl not in last_values or last_values[ctl] != value
    }
    for ctl, value in diff_ctl_values.items():
        port.send(mido.Message("control_change", control=ctl, value=value))
