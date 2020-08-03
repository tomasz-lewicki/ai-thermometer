import cv2

def gstreamer_pipeline(
    capture_width=3264,
    capture_height=2464,
    display_width=1024,
    display_height=768,
    framerate=10,
    flip_method=2,
):

    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink max-buffers=1 drop=true"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


if __name__ == "__main__":
    SIZE = (1024, 768)
    
    cap = cv2.VideoCapture(
        gstreamer_pipeline(
            capture_width=3264,
            capture_height=2464,
            display_width=SIZE[0],
            display_height=SIZE[1],
            framerate=21,
            flip_method=2,
        ),
        cv2.CAP_GSTREAMER,
    )

    for i in range(10):
        ret_val, rgb_arr = cap.read()
        cv2.imshow("show jetson", rgb_arr)

    cap.release()
    cv2.destroyAllWindows()