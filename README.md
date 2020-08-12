# Demo 

## Version 0.3: ResNet-10 Facial Detection + FLIR Lepton 3.5 temp. measurement (~17 FPS w/ GPU accel.)

Video coming soon!

## Version 0.2: Haar Cascade + FLIR Lepton 3.5 temp. measurement (~10FPS, CPU only)
Watch full video: [https://www.youtube.com/watch?v=j9eo9Cs8J8I](https://www.youtube.com/watch?v=j9eo9Cs8J8I)

![](docs/images/haar/combined_short.gif)


## Version 0.1: SSD (COCO) + FLIR Lepton 3.5 temp. measurement (~12FPS w/ GPU accel.)
Watch full video: [https://www.youtube.com/watch?v=i2XMtshdjn8](https://www.youtube.com/watch?v=i2XMtshdjn8)

![](docs/images/ssd/ssd_short.gif)


# Installation 

Make sure you have the newest Nvidia JetPack to avoid issues.

1. Update package manager

```shell
sudo apt update && sudo apt upgrade
```

2. You will need OpenCV 4.4+ for CUDA DNN support. Install it with [this script](https://github.com/mdegans/nano_build_opencv/blob/master/build_opencv.sh):

```shell
wget https://raw.githubusercontent.com/mdegans/nano_build_opencv/master/build_opencv.sh
./build_opencv.sh
```

3. Install AI Thermometer

```shell
git clone https://github.com/tomek-l/ai-thermometer
pip3 install -r requirements.txt
export DISPLAY=:0.0 # (if accessing remotely)
python3 main.py
```

# Possible issues:
- Make sure your user has r/w access to the camera device. 
- the provided  ```libuvc.so``` is compiled for AArch64. It is a custom version of libuvc that supports Y16 video format.
If you're using different architecture, you will need to build the library from source:

```shell
git clone https://github.com/groupgets/libuvc
cd libuvc
mkdir build
cd build
cmake ..
make
cp libuvc.so ~/ai-thermometer/ir/libuvc_wrapper
```


