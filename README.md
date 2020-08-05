# Installation 

Prerequisites:
- OpenCV 4.4+ (you will likely have to compile it yourself)

```
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
