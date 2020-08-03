gst-launch-1.0 v4l2src device=/dev/video1 ! video/x-raw,format=GRAY8 ! videoscale ! video/x-raw,width=1440,height=1080 ! videoconvert ! omxh264enc ! qtmux ! filesink location=IR.mp4 -e
