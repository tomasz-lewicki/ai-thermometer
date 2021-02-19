import numpy as np
import cv2
from utils.transforms import img2euc, euc2img, shift, zoom_out


def make_combined_view(rgb_arr, ir_arr):

    # W, H = rgb_arr.shape[:2]
    a = rgb_arr
    b = ir_arr
    b = cv2.cvtColor(ir_arr, cv2.COLOR_GRAY2BGR)
    # rgb_arr = cv2.resize(b, (W, H))
    arr = cv2.addWeighted(a, 0.5, b, 0.5, 0)

    return arr


def make_rgb_view(arr, scores, boxes, landms, win_size):

    W, H = win_size
    arr = cv2.resize(arr, (W, H))

    for score, box, landm in zip(scores, boxes, landms):

        # convert boxes to pixel frame
        box_px = np.array([W, H, W, H]) * box
        box_px = np.rint(box_px).astype(np.int)
        x1, y1, x2, y2 = box_px

        # draw bounding box
        cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # draw label
        cv2.putText(
            arr,
            f"conf: {score*100:2.0f}%",
            org=(x1, y1 - 10 if y1 > 20 else y1 + 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 0),
            thickness=1,
        )

        if any(landm):
            # convert landmarks to pixel frame
            landm_px = np.array([W, H] * 5) * landm
            landm_px = np.rint(landm_px).astype(np.int)

            # draw landmarks
            cv2.circle(arr, (landm_px[0], landm_px[1]), 1, (0, 255, 0), 2)
            cv2.circle(arr, (landm_px[2], landm_px[3]), 1, (0, 255, 0), 2)
            cv2.circle(arr, (landm_px[4], landm_px[5]), 1, (0, 255, 0), 2)
            cv2.circle(arr, (landm_px[6], landm_px[7]), 1, (0, 255, 0), 2)
            cv2.circle(arr, (landm_px[8], landm_px[9]), 1, (0, 255, 0), 2)

    return arr


def draw_box(arr, box):

    (H, W) = arr.shape[:2]
    box_px = np.array([W, H, W, H]) * box
    box_px = np.rint(box_px).astype(np.int)
    x1, y1, x2, y2 = box_px
    cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 255, 255), 2)


def make_ir_view(
    arr, scores, boxes, landms, temps, calib_box, win_size, thr_min=32, thr_max=40
):

    W, H = win_size
    arr = cv2.resize(arr, (W, H))
    arr = colormap(arr, thr_min, thr_max)

    draw_box(arr, calib_box)

    for score, box, landm, temp in zip(scores, boxes, landms, temps):

        # convert boxes to pixel frame
        box_px = np.array([W, H, W, H]) * box
        box_px = np.rint(box_px).astype(np.int)
        x1, y1, x2, y2 = box_px

        # draw bounding box
        cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # draw temp info
        (Tavg, Tmax, T90th) = temp

        line1 = (x1, y1 - 35 if y1 > 20 else y1 + 10)
        line2 = (line1[0], line1[1] + 15)
        line3 = (line1[0], line1[1] + 30)

        # TODO: there must be a better way to put multiline text..
        cv2.putText(
            arr,
            f"Tavg: {Tavg:2.2f}C",
            org=line1,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
        )

        cv2.putText(
            arr,
            f"Tmax: {Tmax:2.2f}C",
            org=line2,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
        )

        cv2.putText(
            arr,
            f"T90th: {T90th:2.2f}C",
            org=line3,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
        )

        if any(landm):
            # convert landmarks to pixel frame
            landm_px = np.array([W, H] * 5) * landm
            landm_px = np.rint(landm_px).astype(np.int)

            # draw landmarks
            cv2.circle(arr, (landm_px[0], landm_px[1]), 1, (0, 255, 0), 2)
            cv2.circle(arr, (landm_px[2], landm_px[3]), 1, (0, 255, 0), 2)
            cv2.circle(arr, (landm_px[4], landm_px[5]), 1, (0, 255, 0), 2)
            cv2.circle(arr, (landm_px[6], landm_px[7]), 1, (0, 255, 0), 2)
            cv2.circle(arr, (landm_px[8], landm_px[9]), 1, (0, 255, 0), 2)

    return arr


def colormap(arr_temps, thr_min=30, thr_max=40):

    # make background (a 3channel normalized gray image)
    bg = np.clip(arr_temps, -273, thr_min)  # so we use the entire 0-255 range in bg
    bg = cv2.normalize(bg, None, 0, 255, cv2.NORM_MINMAX)
    bg = bg.astype(np.uint8)
    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)  # 1ch -> 3ch

    # make foreground
    fg = np.clip(arr_temps, thr_min, thr_max)

    # this makes the colormap stretched between thr_min and thr_max
    # rather than thr_min and fg.max()
    cmap_max = (fg.max() - thr_min) * 255 / (thr_max - thr_min)
    fg = cv2.normalize(fg, None, 0, cmap_max, cv2.NORM_MINMAX)

    fg = fg.astype(np.uint8)
    fg = cv2.applyColorMap(fg, cv2.COLORMAP_JET)

    mask = np.logical_and(arr_temps > thr_min, arr_temps < thr_max)
    mask = np.stack(3 * [mask], axis=-1)

    return np.where(mask, fg, bg)


def ctof(c):
    f = (c * 9 / 5) + 32
    return f


def draw_rectangle(arr):
    center = np.array([0.5, 0.5])
    wh = np.array([0.5, 0.5])

    p1 = np.array([center[0] - wh[0] / 2, center[1] - wh[1] / 2])

    p2 = np.array([center[0] + wh[0] / 2, center[1] + wh[1] / 2])

    # scale it to image size
    s = np.array(arr.shape[:2][::-1])

    p1 = tuple(np.array(p1 * s, dtype=np.int))
    p2 = tuple(np.array(p2 * s, dtype=np.int))

    cv2.rectangle(arr, p1, p2, (255, 255, 255), 1)


def make_gyr_cmap(temps_arr, thr=[30, 36, 36]):
    """
    make heatmap colorized with blue, yellow and red according to following thresholds
    """
    hmap = np.zeros((*temps_arr.shape[:2], 3), dtype=np.uint8)

    green_mask = np.logical_and(temps_arr > thr[0], temps_arr < thr[1])
    yellow_mask = np.logical_and(temps_arr > thr[1], temps_arr < thr[2])
    red_mask = temps_arr > thr[2]

    hmap[green_mask] = (0, 255, 0)  # green
    hmap[yellow_mask] = (0, 255, 255)  # yellow
    hmap[red_mask] = (0, 0, 255)  # red

    return hmap


def make_bin_cmap(temps_arr, thr=37):
    """
    colorize with red above thr
    """
    hmap = np.zeros((*temps_arr.shape[:2], 3), dtype=np.uint8)
    hmap[temps_arr > thr] = (0, 0, 255)  # red

    return hmap
