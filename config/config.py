import os
import platform

# --------------------------------------------------------
# Automatically find YOLOv2_FACE base directory
# --------------------------------------------------------

def find_base_dir(target_folder='YOLOv2_FACE'):
    cur_path = os.path.abspath(__file__)
    while True:
        cur_path = os.path.dirname(cur_path)
        if os.path.basename(cur_path) == target_folder:
            return cur_path
        if cur_path == os.path.dirname(cur_path):
            raise RuntimeError(f"Cannot find base directory named '{target_folder}'")

BASE_DATA_DIR = find_base_dir()  #auto root.
BASE_DIR = BASE_DATA_DIR

# --------------------------------------------------------
# Dataset / model path settings (OS-independent)
# --------------------------------------------------------

PRETRAINED_MODEL_PATH = os.path.join(BASE_DATA_DIR, "data", "pretrained", "darknet19_448.weights")
CHECKPOINT_DIR = os.path.join(BASE_DATA_DIR, "output")
WIDERFACE_IMAGES = os.path.join(BASE_DATA_DIR, "images")
WIDERFACE_ANNOTATIONS = os.path.join(BASE_DATA_DIR, "Annotations")
WIDERFACE_IMAGESETS = os.path.join(BASE_DATA_DIR, "ImageSets", "Main")
LABELS_DIR = os.path.join(BASE_DATA_DIR, "labels")

# --------------------------------------------------------
# Model and training configuration
# --------------------------------------------------------

#VOC 2007+2012 dataset
anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
#anchors = [[0.4233, 0.5351], [1.4464, 1.8470], [3.5525, 4.7339], [7.7083, 10.2665], [10.1851, 12.5]]
#13fit
#anchors = [[0.4233, 0.5351], [1.4464, 1.8470], [3.5525, 4.7339], [7.7083, 10.2665], [9.8534, 13.0000]]
#anchors = [[0.4233, 0.5351], [1.4464, 1.8470], [3.5525, 4.7339], [7.7083, 10.2665], [14.5502, 19.1967]]
#face+face_new --> 거의 1:1.3
#anchors = [[1.0042, 1.3086], [3.1126, 4.2410], [6.1695, 8.1148], [9.7812, 13.0000], [9.9573, 13.0000]]


object_scale = 5
noobject_scale = 1
class_scale = 1
coord_scale = 1

saturation = 1.5
exposure = 1.5
hue = .1

jitter = 0.3

thresh = .6

batch_size = 16

lr = 0.0001

decay_lrs = {
    60: 0.00001,
    90: 0.000001
}

momentum = 0.9
weight_decay = 0.0005


# multi-scale training:
# {k: epoch, v: scale range}
multi_scale = True

# number of steps to change input size
scale_step = 40

scale_range = (3, 4)

epoch_scale = {
    1:  (3, 4),
    15: (2, 5),
    30: (1, 6),
    60: (0, 7),
    75: (0, 9)
}

input_sizes = [(320, 320),
               (352, 352),
               (384, 384),
               (416, 416),
               (448, 448),
               (480, 480),
               (512, 512),
               (544, 544),
               (576, 576)]

input_size = (416, 416)

test_input_size = (416, 416)

strides = 32

debug = False

