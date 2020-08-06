import numpy as np
import cv2

def draw_boxes(arr, detections):
    
    if detections is None:
        return arr

    h, w = arr.shape[:2]
    scores = detections[:,2]
    
    # loop over the detections
    for (_, _, score, x1, y1, x2, y2) in detections[scores > 0.5]:
        
        # scale box
        box = np.array([x1, y1, x2, y2]) * np.array([w, h, w, h])
        
        # cast to int
        (x1, y1, x2, y2) = box.astype("int")
        
        # draw box
        cv2.rectangle(arr, (x1, y1), (x2, y2),(0, 0, 255), 2)

        # put text
        cv2.putText(
            arr,
            f"{score*100:.2f}",
            (x1, y1 - 10 if y1 > 20 else y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2
        )

    return arr