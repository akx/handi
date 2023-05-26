import cv2


def get_cam_frame(cam):
    h = 600
    ok, frame = cam.read()
    if not ok:
        raise RuntimeError("Could not read from camera")
    ar = frame.shape[1] / frame.shape[0]
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (int(h * ar), h))
    return frame
