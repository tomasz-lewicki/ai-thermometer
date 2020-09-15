import numpy as np
import cv2
from .transforms import img2euc, euc2img, shift


def make_rgb_view(arr, detections):

    if detections is None:
        return arr

    h, w = arr.shape[:2]
    scores = detections[:, 2]

    # loop over the detections
    for (_, _, score, x1, y1, x2, y2) in detections[scores > 0.2]: #TODO: put the thresholding outside 

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

def normalize_ir(ir_arr):
    ir_arr_normalized = cv2.normalize(ir_arr, None, 0, 255, cv2.NORM_MINMAX)  # normalize to 0-255
    ir_arr_normalized = ir_arr_normalized.astype(np.uint8)  # convert to uint8
    return ir_arr_normalized

def apply_cmap(ir_arr, threshold=36):

    ir_arr_n = normalize_ir(ir_arr)
    ir_arr_n = ir_arr_n.astype(np.uint8)
    arr_3ch = cv2.cvtColor(ir_arr_n, cv2.COLOR_GRAY2BGR)

    ir_arr_n = cv2.applyColorMap(ir_arr_n, cv2.COLORMAP_JET)

    mask = ir_arr<threshold
    mask = np.stack(3*[mask], axis=-1)

    return np.where(mask, arr_3ch, ir_arr_n)


def make_ir_view(rgb_arr, ir_arr, detections, temperatures):

    # scaling width, scaling height
    SW, SH = 0.5, 0.5

    if detections is None:
        return ir_arr

    h_rgb, w_rgb = rgb_arr.shape[:2]
    h_ir, w_ir = ir_arr.shape[:2]

    scores = detections[:, 2]

    ir_arr_3ch = apply_cmap(temperatures)

    # loop over the detections
    for (_, _, score, x1, y1, x2, y2) in detections[scores > 0.2]: #TODO: adjustable threshold

        # TODO: interface should only receive the detections after transform!
        # box in euclidean rgb frame
        x1, y1 = img2euc(x1, y1)
        x2, y2 = img2euc(x2, y2)

        # scale to convert to ir frame
        x1 /= SW
        x2 /= SW
        y1 /= SH
        y2 /= SH

        # shift left
        x1, y1 = shift(x1, y1, +0.05, 0)
        x2, y2 = shift(x2, y2, +0.05, 0)

        # convert back to image frame
        x1, y1 = euc2img(x1, y1)
        x2, y2 = euc2img(x2, y2)

        # scale
        box = np.array([x1, y1, x2, y2])
        box *= np.array([w_ir, h_ir, w_ir, h_ir])

        # cast to int
        (x1, y1, x2, y2) = box.astype("int")

        # draw box
        cv2.rectangle(ir_arr_3ch, (x1, y1), (x2, y2), (0, 0, 255), 2)

        roi = temperatures[y1:y2, x1:x2]
        temp = np.nanmean(roi[roi > 32])

        # put text
        cv2.putText(
            ir_arr_3ch,
            f"{temp:.2f} C",
            (x1, y1 - 10 if y1 > 20 else y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
        )

    return ir_arr_3ch
