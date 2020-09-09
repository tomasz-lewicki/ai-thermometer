import numpy as np
import cv2


def draw_boxes(arr, detections):

    if detections is None:
        return arr

    h, w = arr.shape[:2]
    scores = detections[:, 2]

    # loop over the detections
    for (_, _, score, x1, y1, x2, y2) in detections[scores > 0.5]:

        # scale box
        box = np.array([x1, y1, x2, y2]) * np.array([w, h, w, h])

        # cast to int
        (x1, y1, x2, y2) = box.astype("int")

        # draw box
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # put text
        cv2.putText(
            arr,
            f"{score*100:.2f}",
            (x1, y1 - 10 if y1 > 20 else y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
        )

    return arr


def img2euc(x, y):
    """
    image frame to Euclidean frame
    """
    x = x - 0.5
    y = -y + 0.5
    return x, y


def euc2img(x, y):
    """
    Euclidean frame to image frame
    """
    x = x + 0.5
    y = -y + 0.5
    return x, y


def shift(x, y, dx, dy):
    x += dx
    y += dy
    return x, y


def overlay_temps(rgb_arr, ir_arr, detections, temperatures):

    # scaling width, scaling height
    SW, SH = 0.5, 0.5

    if detections is None:
        return ir_arr

    h_rgb, w_rgb = rgb_arr.shape[:2]
    h_ir, w_ir = ir_arr.shape[:2]

    scores = detections[:, 2]

    # loop over the detections
    for (_, _, score, x1, y1, x2, y2) in detections[scores > 0.5]:

        # box in euclidean rgb frame
        x1, y1 = img2euc(x1, y1)
        x2, y2 = img2euc(x2, y2)

        # scale to convert to ir frame
        x1 /= SW
        x2 /= SW
        y1 /= SH
        y2 /= SH

        # shift right
        x1, y1 = shift(x1, y1, -0.075, 0)
        x2, y2 = shift(x2, y2, -0.075, 0)

        # convert back to image frame
        x1, y1 = euc2img(x1, y1)
        x2, y2 = euc2img(x2, y2)

        # scale
        box = np.array([x1, y1, x2, y2])
        box *= np.array([w_ir, h_ir, w_ir, h_ir])

        # cast to int
        (x1, y1, x2, y2) = box.astype("int")

        # draw box
        cv2.rectangle(ir_arr, (x1, y1), (x2, y2), (0, 0, 255), 2)

        roi = temperatures[x1:x2, y1:y2]
        temp = np.mean(roi[roi > 32])

        # put text
        cv2.putText(
            ir_arr,
            f"{temp:.2f} C",
            (x1, y1 - 10 if y1 > 20 else y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
        )

    return ir_arr
