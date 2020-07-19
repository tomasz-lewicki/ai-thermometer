gst-launch-1.0 nvarguscamerasrc sensor_mode=0 ! 'video/x-raw(memory:NVMM),width=3264, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=2  ! nvvidconv top=96 bottom=2368 left=137 right=3127 ! 'video/x-raw(memory:NVMM),width=1440, height=1080' ! omxh264enc ! qtmux ! filesink location=RGB.mp4 -e

