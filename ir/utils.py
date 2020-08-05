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

