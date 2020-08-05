# Demo (click on images to watch the video)

## Version 0.3: ResNet-10 Facial Detection + FLIR Lepton 3.5 temp. measurement (~17 FPS w/ GPU accel.)
Coming soon.

## Version 0.2: Haar Cascade + FLIR Lepton 3.5 temp. measurement (~10FPS, CPU only)
[![Watch the video](https://img.youtube.com/vi/j9eo9Cs8J8I/hqdefault.jpg)](https://www.youtube.com/watch?v=j9eo9Cs8J8I)


## Version 0.1: SSD (COCO) + FLIR Lepton 3.5 temp. measurement (~12FPS w/ GPU accel.)
Click to watch the video:

[![Watch the video](https://img.youtube.com/vi/i2XMtshdjn8/maxresdefault.jpg)](https://www.youtube.com/watch?v=i2XMtshdjn8)




# Installation 

Prerequisites:
- OpenCV 4.4+ (you will likely have to compile it yourself)

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


