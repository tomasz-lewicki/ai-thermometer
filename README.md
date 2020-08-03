

# Installation (detailed)

First, we will require a custom libuvc from https://github.com/groupgets/libuvc

1. Compile the modified libuvc

```shell
git clone https://github.com/groupgets/libuvc
cd libuvc
mkdir build
cd build
cmake ..
make
```

Copy the resulting __libuvc.so__ file to this repo
```
cp ~/tmp/libuvc/build/libuvc.so ~/crowd-thermometer/
```

2. Install OpenCV
```
sudo apt-get install python-opencv
```

3. Install ROS2

https://index.ros.org/doc/ros2/Installation/

4. Install jetson-inference

https://github.com/dusty-nv/jetson-inference

5. Make sure your user has r/w access to the camera. 

For a quick and dirty solution do:

```shell
echo 'SUBSYSTEM=="usb",  ENV{DEVTYPE}=="usb_device", GROUP="plugdev", MODE="0664"' | sudo tee /etc/udev/rules.d/10-libuvc.rules 
sudo udevadm trigger
```
<span style="color: red">(this gives camera access to all users)</span>

For a better, fine-grained permission setup, create a new group, give permissions to that group only and your user to the newly created group
To read more: http://wiki.ros.org/libuvc_camera#Permissions

7. Run the nodes
```shell
source /opt/ros/dashing/setup.bash
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

source /opt/ros/dashing/setup.bash
