HZ_CAP = 20
LOG_DIR = "logs"
VIS_WIN_NAME = "RGB view"
IR_WIN_NAME = "IR view"

VIS_BBOX_COLOR = (0, 0, 255)  # red
IR_BBOX_COLOR = (0, 255, 0)  # green

IR_WIN_SIZE = (960, 720)  # splits 1080p screen in half
VIS_WIN_SIZE = (960, 720)

SAVE_FRAMES = True
SHOW_DISPLAY = True
MAX_FILE_QUEUE = 10

X_DISPLAY_ADDR = ":0"

FACE_DET_MODEL = "retinaface"  # alternatively SSD

CALIBRATE = False # We default to false. Otherwise very large errors for users who deploy without a BB reference.
CALIB_T = 40 # temperature to which the blackbody reference is set to
CALIB_BOX = [8/160, 106/120, 20/160, 115/120]

CMAP_TEMP_MIN = 30
CMAP_TEMP_MAX = 40