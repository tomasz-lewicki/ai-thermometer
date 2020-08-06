
import numpy as np
import time
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

def draw_boxes(arr, detections):
	# loop over the detections
	h, w = arr.shape[:2]
	scores = detections[:,2]

	for (_, _, score, x1, y1, x2, y2) in detections[scores > 0.5]:
		
		# scale box
		box = np.array([x1, y1, x2, y2]) * np.array([w, h, w, h])
		
		# cast to int
		(x1, y1, x2, y2) = box.astype("int")
		
		# draw box
		cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 0, 255), 2)

		# put text
		cv2.putText(
			frame,
			f"{score*100:.2f}",
			(x1, y1 - 10 if y1 > 20 else y1 + 10),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(0, 0, 255),
			2
		)

if __name__ == "__main__":
	net = cv2.dnn.readNetFromCaffe("caffe/deploy.prototxt.txt", "caffe/res10_300x300_ssd_iter_140000.caffemodel")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	stream = cv2.VideoCapture(
		gstreamer_pipeline(
			capture_width=3264,
			capture_height=2464,
			display_width=1024,
			display_height=768,
			framerate=21,
			flip_method=2,
		),
		cv2.CAP_GSTREAMER,
	)

	while True:

		loop_start = time.monotonic()
		ret, frame = stream.read()

		# detect
		blob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)),
			1.0,
			(300, 300),
			(104.0, 177.0, 123.0)
			)
		net.setInput(blob)
		detections = net.forward()

		# overlay
		detections = np.squeeze(detections)
		draw_boxes(frame, detections)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		print(f"delay {1000*(time.monotonic()-loop_start):.1f} ms")

	# cleanup
	cv2.destroyAllWindows()
	stream.stop()

