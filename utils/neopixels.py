import board
import time
import neopixel_spi as neopixel

NUM_PIXELS = 8

NEOPIXEL_COLORS = {
    "RED": 0xFF0000,
    "GREEN": 0x00FF00,
    "BLUE": 0x0000FF,
    "BLACK": 0x000000,
    # "WHITE": 0xFFFFFF,
}


class NeopixelIndicator:
    def __init__(self):
        spi = board.SPI()

        self._pixels = neopixel.NeoPixel_SPI(
            spi, n=NUM_PIXELS, pixel_order=neopixel.GRBW, auto_write=True
        )

    def set_color(self, color):
        # color=0xFF0000 for red
        self._pixels.fill(color)


if __name__ == "__main__":

    indicator = NeopixelIndicator()

    try:
        while True:
            for c in NEOPIXEL_COLORS.values():
                indicator.set_color(c)
                time.sleep(0.3)
    except KeyboardInterrupt:
        indicator.set_color(NEOPIXEL_COLORS["BLACK"])
