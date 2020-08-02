
from threading import Thread
from queue import Queue

import cv2
import time

from ir import IRThread
from vis import GPUThread

def exit_handler():
    print("exit handler called")
    gpu_thread.stop()
    ir_thread.stop()
    
    gpu_thread.join()
    ir_thread.join()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    ir_thread = IRThread()
    ir_thread.start()

    gpu_thread = GPUThread()
    gpu_thread.start()

    try:
        while True:
            time.sleep(.05)

            if gpu_thread._frame is None:
                print("Waiting for frames")
                time.sleep(1)
                continue
            else:
                print(f"GPU thread delay={gpu_thread._delay:.2f}")
            
            cv2.imshow("VIS frame", gpu_thread.frame)
            cv2.imshow("IR frame", ir_thread.frame)
            key = cv2.waitKey(1) & 0xFF
            
            # if the `q` key was pressed in the cv2 window, break from the loop
            if key == ord("q"):
                break

    finally:
        exit_handler()



    # ct = CameraThread(stream)

    # ct.start()


