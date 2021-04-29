# AI Thermometer
Contactless temperature mesurement using IR & RGB cameras and Deep CNN facial detection.

### Normal temperature
![](docs/images/retinaface/healthy.gif)
### Elevated temperature
![](docs/images/retinaface/elevated.gif)

## What's new in v0.4
* New facial detector (handles partial occlusion, e.g. with a mask)
* Added IR camera drift compensation with an external blackbody

## Hardware 
To build this project you will need:
|#|Part|link|Price (USD)|
|-|-|-|-|
|1|Jetson Nano Dev Kit|[link](https://www.sparkfun.com/products/16271)| 99 | 
|3|FLIR Lepton 3.5 IR Camera | [link](https://store.groupgets.com/products/flir-lepton-3-5)| 199 |
|4|GroupGets Purethermal2 Module | [link](https://store.groupgets.com/products/purethermal-2)| 99 |
|2|Raspberry Pi Camera Module V2.1 |[link](https://www.sparkfun.com/products/14028) | 25 |
|5|Noctua cooling fan | [link](https://www.amazon.com/Noctua-NF-A4x10-PWM-4-Pin-Premium/dp/B07DXRNYNX/) | 14 |
|6|3D printed enclosure | [3D model](https://a360.co/3g8kfqV) | - |
|total | | | 436 |
![wireframe image](docs/images/wireframe.png)

## Quickstart 

1. Flash Nvidia Jetson Nano with the latest Nvidia JetPack. Update package manager

```shell
sudo apt update && sudo apt upgrade
```

2. Build OpenCV >4.4. It is required for CUDA DNN support.
```shell
wget https://raw.githubusercontent.com/mdegans/nano_build_opencv/master/build_opencv.sh
chmod +x build_opencv.sh
./build_opencv.sh
```
3. Build pytorch (instructions [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048))

4. Download AI Thermometer

```shell
git clone https://github.com/tomek-l/ai-thermometer
cd ai-thermometer
```

5. Run AI Thermometer

```shell
python3 main.py
```

6. Optionally, 3D print & assemble the enclosure from [here](https://a360.co/3g8kfqV)

## Limitations

### IR and RGB camera alignment.

The current way of calculating the correspondence between IR and RGB cameras is not ideal.
Factors, such as the non-rigid mount of the sensor on the Raspberry Pi CMV2.1 don't help.
I'm actively working on calibration code that takes into account the intrinsic parameters of both cameras, which should allow for obtaining a pixel-level correspondence between the imags.

## FAQ/common issues:

1. `uvc_open error -3`

Reason: Your current user does not have r/w access to the PureThermal USB device. 

For a quick and dirty fix you can do do:
```shell
echo 'SUBSYSTEM=="usb",  ENV{DEVTYPE}=="usb_device", GROUP="plugdev", MODE="0664"' | sudo tee /etc/udev/rules.d/10-libuvc.rules 
sudo udevadm trigger
```
(this gives camera access to all users)

For a better, fine-grained permission setup, create a new group, give permissions to that group only and your user to the newly created group. You can read more [here](http://wiki.ros.org/libuvc_camera#Permissions).

2. `Illegal instruction (core dumped)`

Reason: The provided  ```libuvc.so``` is compiled for AArch64.
It is a custom version of libuvc that supports Y16 video format. If you're using different architecture, you will need to build the library from source:

```shell
git clone https://github.com/groupgets/libuvc
cd libuvc
mkdir build
cd build
cmake ..
make
cp libuvc.so ~/ai-thermometer/ir/libuvc_wrapper
```

3. using sudo or getting `permission denied`

You don't need ```sudo``` to run the code in this repository 🙂

## Previous versions

## Version 0.2: Haar Cascade + FLIR Lepton 3.5 temp. measurement (~10FPS, CPU only)
Watch full video: [https://www.youtube.com/watch?v=j9eo9Cs8J8I](https://www.youtube.com/watch?v=j9eo9Cs8J8I)

![](docs/images/haar/combined_short.gif)


## Version 0.1: SSD (COCO) + FLIR Lepton 3.5 temp. measurement (~12FPS w/ GPU accel.)
Watch full video: [https://www.youtube.com/watch?v=i2XMtshdjn8](https://www.youtube.com/watch?v=i2XMtshdjn8)

![](docs/images/ssd/ssd_short.gif)

