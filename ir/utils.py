import cv2
import numpy as np

def ktoc(val):
    # Kelvin to Celsius
    return (val - 27315) / 100.0

def resize(ir_arr, size):
    ir_arr = cv2.resize(ir_arr, size)
    return ir_arr

def normalize(ir_arr):
    ir_arr_n = cv2.normalize(ir_arr, None, 0, 255, cv2.NORM_MINMAX)  # normalize to 0-255
    ir_arr_n = ir_arr_n.astype(np.uint8)  # convert to uint8
    return ir_arr_n

def crop_telemetry(ir_arr):
    return ir_arr[:-2,:]

def detect_ir(ir_arr, thr):
    """
    Detects objects above thr temperature in ir array
    :param ir_arr: ir array in deg. C
    :param thr: threshold temperature in deg. C
    """
    
    mask = ir_arr>thr
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for c in contours:
        polygon = cv2.approxPolyDP(c, 3, True)
        bbox = cv2.boundingRect(polygon)
        bboxes.append(bbox)
    
    return bboxes

def drop_small_bboxes(bboxes, min_size):
    """
    :param min_size: min size of bb area [px]
    """
    good = []
    for (x, y, dx, dy) in bboxes:
        if dx*dy > min_size:
            good.append((x, y, dx, dy))
    return good

def overlay_bboxes(arr, bboxes):
    
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    arr = arr.astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    
    COL = (255, 255, 0)
    for (x, y, dx, dy) in bboxes:
        p1 = (x,y)
        p2 = (x+dx, y+dy)
        cv2.rectangle(arr,
                      p1,
                      p2,
                      color=COL,
                      thickness=2)
    return arr