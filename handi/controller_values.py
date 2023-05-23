import mido

from handi.hand_result import HandResult


def compute_ctl_values(hand: HandResult):
    pos_base_ctl = 18
    ang_base_ctl = 32
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
    return ctl_values


def send_changed_values(port, last_values, new_values):
    diff_ctl_values = {
        ctl: value
        for ctl, value in new_values.items()
        if ctl not in last_values or last_values[ctl] != value
    }
    print(f"{diff_ctl_values=}")
    for ctl, value in diff_ctl_values.items():
        port.send(mido.Message("control_change", control=ctl, value=value))