# IR + RGB see

gst-launch-1.0 nvarguscamerasrc sensor_mode=0 ! 'video/x-raw(memory:NVMM),width=3264, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=2 ! 'video/x-raw,width=800, height=600' ! videoconvert ! ximagesink & gst-launch-1.0 v4l2src device=/dev/video1 ! video/x-raw,format=UYVY ! videoscale ! video/x-raw,width=800,height=600 ! videoconvert ! ximagesink


# RGB

## Capture at full resolution, display on the screen
```shell
gst-launch-1.0 nvarguscamerasrc sensor_mode=0 ! 'video/x-raw(memory:NVMM),width=3264, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=2 ! 'video/x-raw,width=960, height=720' ! nvvidconv ! nvegltransform ! nveglglessink -e
```

## Cropped
```shell
gst-launch-1.0 nvarguscamerasrc sensor_mode=0 ! 'video/x-raw(memory:NVMM),width=3264, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=2 ! videocrop top=96 left=137 right=137 bottom=96 ! 'video/x-raw,width=800, height=600' ! nvvidconv ! nvegltransform ! nveglglessink -e
```

## General form to add cropping:
```
gst-launch-1.0 -v videotestsrc ! videocrop top=96 left=160 right=160 bottom=96 ! ximagesink
```

## Nvidia overlay display
```
gst-launch-1.0 nvarguscamerasrc sensor_mode=0 ! 'video/x-raw(memory:NVMM),width=3264, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=2  ! nvvidconv top=100 bottom=2364 left=300 right=3164 ! 'video/x-raw(memory:NVMM),width=1440, height=1080' ! nvoverlaysink display-id=0 -e
```

## take away 137 on the sides, 

horizontal: 3264-137=3127
vertical: 2464-96=2368
```
gst-launch-1.0 nvarguscamerasrc sensor_mode=0 ! 'video/x-raw(memory:NVMM),width=3264, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=2  ! nvvidconv top=96 bottom=2368 left=137 right=3127 ! 'video/x-raw(memory:NVMM),width=1440, height=1080' ! omxh264enc ! qtmux ! filesink location=RGB.mp4 -e 
```

## Saving to files
Save 1080p (sensor sub-frame)
```shell
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1920, height=1080,format=NV12, framerate=30/1'  ! nvvidconv flip-method=2 ! omxh264enc ! qtmux ! filesink location=1080p_output_cropped.mp4 -e
```

Save 1080p (from full frame, then resize)
```shell
gst-launch-1.0 nvarguscamerasrc sensor_mode=0 ! 'video/x-raw(memory:NVMM),width=3264, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=2 ! 'video/x-raw,width=1440, height=1080' ! omxh264enc ! qtmux ! filesink location=1080p_from_full.mp4 -e
```

--------------------------------------------------------------------------------------------------------------------------------------------------------

# Capture from IR Lepton Camera

## IR camera display (w/ colormap)
```shell
gst-launch-1.0 v4l2src device=/dev/video1 ! video/x-raw,format=UYVY ! videoscale ! video/x-raw,width=800,height=600 ! videoconvert ! ximagesink
```

## IR camera display (RAW 16bit values)
```shell
gst-launch-1.0 v4l2src device=/dev/video1 ! video/x-raw,format=GRAY16_LE ! videoscale ! video/x-raw,width=800,height=600 ! videoconvert ! ximagesink
```

## IR  - Save 

### Colormap to jpeg
```
gst-launch-1.0 v4l2src device=/dev/video1 num-buffers=1 ! video/x-raw,format=UYVY ! videoscale ! video/x-raw,width=800,height=600 ! nvjpegenc ! filesink location=test_ir.jpg
```

### Gray 16-bit 160x120 to mp4
```shell
gst-launch-1.0 v4l2src device=/dev/video1 ! video/x-raw,format=GRAY16_LE ! videoscale ! video/x-raw,width=160,height=120 ! videoconvert ! omxh264enc ! qtmux ! filesink location=grayscale_raw.mp4 -e
```
## Save to avi
gst-launch-1.0 v4l2src device=/dev/video1 ! video/x-raw,format=GRAY16_LE ! videoscale ! video/x-raw,width=800,height=600 ! videoconvert ! jpegenc ! avimux ! filesink location=test.avi -e

## Gray 8-bit resized to 800x600
gst-launch-1.0 v4l2src device=/dev/video1 ! video/x-raw,format=GRAY8 ! videoscale ! video/x-raw,width=800,height=600 ! videoconvert ! omxh264enc ! qtmux ! filesink location=grayscale_raw.mp4 -e


To record footage from both IR and RGB cameras at the same time, do:
```shell
bash deploy_rgb.sh & bash deploy_ir.sh ; fg
```

To combine the footage, do:
```shell
bash combine_ir_rgb.sh
```

