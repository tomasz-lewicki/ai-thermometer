import numpy as np
import cv2

def ktoc(val):
    # Kelvin to Celsius
    return (val - 27315) / 100.0

def vis2arr(img,size=(1024, 768), to_gray=False):
    
    arr = np.array(img)
    if size:
        arr = cv2.resize(arr, (1024, 768))
    if to_gray:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr

def ir2arr(img, size=(1024, 768)):

    arr = np.array(img, dtype=np.float32)[:-2, :] # trim the 2 bottom lines
    arr = cv2.resize(ktoc(arr), (1024, 768))
    
    return arr

def normalize_ir(arr):
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX) # deg. C to 0-255
    return arr.astype(np.uint8)
