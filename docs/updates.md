# Hardware V2 update

## Camera Upgrade
The RGB camera module has been upgraded from the `Raspberry Pi CM V2.1` to a `Sainsmart IMX219`. 

The main reason for that change is the ridgidity of the assembly.
Specifically, the sensor on the CM V2.1 is held in place on the PCB with a piece of double-sided adhesive.
Even a sub-milimeter shift to the camera position can tilt the camera plane by a few degrees, and mess up the calibration.
That's bad for any machine-vision application.

In the Sainsmart camera module, the sensor is soldered to the PCB, and the plastic lens assembly is screwed down to the board. This is a much more ridgid assembly, making the RGB-IR alignment more robust.

## Antennas


# Software Update
- Support multiple detectors (SSD, retinaface)
