import numpy as np
import cv2


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
    """
    shift x,y by (dx,dy) vector
    """
    x += dx
    y += dy
    return x, y


def zoom_out(arr):
    W, H = arr.shape[:2]

    padded = cv2.copyMakeBorder(
        arr, 125, 75, 75, 75, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    zoomed_out = cv2.resize(padded, (H, W), interpolation=cv2.INTER_CUBIC)

    return zoomed_out


def transform_boxes(bboxes, scale_x=1, scale_y=1, shift_x=0, shift_y=0):

    bboxes_out = []

    for i, bbox in enumerate(bboxes):

        x1, y1, x2, y2 = bbox

        # img -> euclidean
        x1, y1 = img2euc(x1, y1)
        x2, y2 = img2euc(x2, y2)

        # scale
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        # shift left
        x1, y1 = shift(x1, y1, shift_x, shift_y)
        x2, y2 = shift(x2, y2, shift_x, shift_y)

        # convert back to image frame
        x1, y1 = euc2img(x1, y1)
        x2, y2 = euc2img(x2, y2)

        bboxes_out.append([x1, y1, x2, y2])

    return np.array(bboxes_out)
