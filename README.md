To run a basic visualization (without any radiometry) you can do:

```shell
gst-launch-1.0 v4l2src device=/dev/video1 ! video/x-raw,format=UYVY ! videoscale ! video/x-raw,width=800,height=600 ! videoconvert ! ximagesink
```


To record footage from both IR and RGB cameras at the same time, do:
```shell
bash deploy_rgb.sh & bash deploy_ir.sh ; fg
```

To combine the footage, do:
```shell
bash combine_ir_rgb.sh
```


However, the stock uvc driver will not allow to access the raw data from the camera. We will require a custom libuvc from https://github.com/groupgets/libuvc

1. Compile the modified libuvc

```shell
git clone https://github.com/groupgets/libuvc
cd libuvc
mkdir build
cd build
cmake ..
make
```

2. Copy the resulting __libuvc.so__ file to this repo
```
cp ~/tmp/libuvc/build/libuvc.so ~/crowd-thermometer/
```

3. Install OpenCV
```
sudo apt-get install python-opencv
```



# Potential issues:

1. __X server BadDrawable error__
```shell
X Error: BadDrawable (invalid Pixmap or Window parameter) 9
  Major opcode: 62 (X_CopyArea)
  Resource id:  0x7800010
```
__Solution__
```shell
export QT_X11_NO_MITSHM=1
```

2. /dev/video* permissions

