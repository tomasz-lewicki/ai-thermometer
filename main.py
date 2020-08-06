
from threading import Thread
from queue import Queue

import cv2
import time

from ir import IRThread
from vis import GPUThread

from ir.utils import overlay_bboxes as overlay_ir_bboxes 
from vis.utils import draw_boxes as overlay_vis_bboxes 

def exit_handler():
    print("exit handler called")
    gpu_thread.stop()
    ir_thread.stop()
    
    gpu_thread.join()
    ir_thread.join()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":


    gpu_thread = GPUThread()
    gpu_thread.start()

    ir_thread = IRThread()
    ir_thread.start()

    try:
        while gpu_thread.frame is None:
            print("Waiting for RGB frames")
            time.sleep(1)

        while ir_thread.frame is None:
            print("Waiting for IR frames")
            time.sleep(1)

        # main loop
        while True:
            time.sleep(.04) # run main thread at ~20Hz

            print(f"GPU thread latency={gpu_thread._delay:.2f}")


            ir_frame_w_overlay = overlay_ir_bboxes(ir_thread.frame, ir_thread.bboxes)
            vis_frame_w_overlay = overlay_vis_bboxes(gpu_thread.frame, gpu_thread.detections)

            # Show
            cv2.imshow("VIS frame", vis_frame_w_overlay)
            cv2.imshow("IR frame", ir_frame_w_overlay)
            key = cv2.waitKey(1) & 0xFF
            
            # if the `q` key was pressed in the cv2 window, break from the loop
            if key == ord("q"):
                break

    finally:
        exit_handler()




