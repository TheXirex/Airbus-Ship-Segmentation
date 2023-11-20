TRAIN_DIR = 'data/train_v2/'
TEST_DIR = 'data/test_v2/'
DATASET_DIR = 'data/train_ship_segmentations_v2.csv'

TEST_IMAGES = ['0e0fb75c9.jpg', '00dc34840.jpg', '00aa79c47.jpg', '0fb27144a.jpg']

BATCH_SIZE = 48
EDGE_CROP = 16
IMG_SCALING = (3, 3)
VALID_IMG_COUNT = 900
MAX_TRAIN_STEPS = 5
MAX_TRAIN_EPOCHS = 100
SAMPLES_PER_GROUP = 4000

ALPHA = 0.8
GAMMA = 2

IMG_DATA_GEN_ARGS = dict(
    featurewise_center=False,
    samplewise_center=False,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.01,
    zoom_range=[0.9, 1.25],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    data_format='channels_last'
)

COLORS = [
    (0, 150, 255),
    (0, 180, 255),
    (0, 200, 255),
    (0, 220, 255),
    (0, 240, 255),
    (0, 255, 240),
    (0, 255, 220),
    (0, 255, 200),
    (0, 255, 180),
    (0, 255, 150),
]