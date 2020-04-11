

gst-launch-1.0 nvarguscamerasrc sensor_mode=0 num-buffers=1 ! 'video/x-raw(memory:NVMM),width=3264, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=2 ! nvjpegenc ! filesink location=test_full.jpg
gst-launch-1.0 nvarguscamerasrc sensor_mode=0 num-buffers=1 ! 'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=21/1, format=NV12' ! nvvidconv flip-method=2 ! nvjpegenc ! filesink location=test_1080p.jpg
gst-launch-1.0 nvarguscamerasrc sensor_mode=0 num-buffers=1 ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=21/1, format=NV12' ! nvvidconv flip-method=2 ! nvjpegenc ! filesink location=test_720p.jpg


gst-launch-1.0 nvarguscamerasrc sensor_mode=0 num-buffers=1 ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=21/1, format=NV12' ! nvvidconv flip-method=2  ! 'video/x-raw,width=1440, height=1080' ! nvjpegenc ! filesink location=test_1080_4by3.jpg

gst-launch-1.0 v4l2src device=/dev/video1 num-buffers=1 ! video/x-raw,format=UYVY ! videoscale ! video/x-raw,width=3264,height=2464 ! nvjpegenc ! filesink location=test_ir.jpg
