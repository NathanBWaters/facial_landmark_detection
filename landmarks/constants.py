'''
Contains our constant / global variables
'''
import os

# make sure our experiments are reproducible
SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATADIR = os.path.join(ROOT, 'facial-keypoints-detection')
TRAINING_DATA = os.path.join(DATADIR, 'training.csv')
TEST_DATA = os.path.join(DATADIR, 'test.csv')
MODEL_DIR = os.path.join(ROOT, 'model_checkpoints')

INPUT_DIMS = (96, 96, 1)

LABEL_TO_ID = {
    'left_eye_center_x': 0,
    'left_eye_center_y': 1,
    'right_eye_center_x': 2,
    'right_eye_center_y': 3,
    'left_eye_inner_corner_x': 4,
    'left_eye_inner_corner_y': 5,
    'left_eye_outer_corner_x': 6,
    'left_eye_outer_corner_y': 7,
    'right_eye_inner_corner_x': 8,
    'right_eye_inner_corner_y': 9,
    'right_eye_outer_corner_x': 10,
    'right_eye_outer_corner_y': 11,
    'left_eyebrow_inner_end_x': 12,
    'left_eyebrow_inner_end_y': 13,
    'left_eyebrow_outer_end_x': 14,
    'left_eyebrow_outer_end_y': 15,
    'right_eyebrow_inner_end_x': 16,
    'right_eyebrow_inner_end_y': 17,
    'right_eyebrow_outer_end_x': 18,
    'right_eyebrow_outer_end_y': 19,
    'nose_tip_x': 20,
    'nose_tip_y': 21,
    'mouth_left_corner_x': 22,
    'mouth_left_corner_y': 23,
    'mouth_right_corner_x': 24,
    'mouth_right_corner_y': 25,
    'mouth_center_top_lip_x': 26,
    'mouth_center_top_lip_y': 27,
    'mouth_center_bottom_lip_x': 28,
    'mouth_center_bottom_lip_y': 29,
}

# Reverse map - I'll be honest this is pretty lame
ID_TO_LABEL = {
    0: 'left_eye_center_x',
    1: 'left_eye_center_y',
    2: 'right_eye_center_x',
    3: 'right_eye_center_y',
    4: 'left_eye_inner_corner_x',
    5: 'left_eye_inner_corner_y',
    6: 'left_eye_outer_corner_x',
    7: 'left_eye_outer_corner_y',
    8: 'right_eye_inner_corner_x',
    9: 'right_eye_inner_corner_y',
    10: 'right_eye_outer_corner_x',
    11: 'right_eye_outer_corner_y',
    12: 'left_eyebrow_inner_end_x',
    13: 'left_eyebrow_inner_end_y',
    14: 'left_eyebrow_outer_end_x',
    15: 'left_eyebrow_outer_end_y',
    16: 'right_eyebrow_inner_end_x',
    17: 'right_eyebrow_inner_end_y',
    18: 'right_eyebrow_outer_end_x',
    19: 'right_eyebrow_outer_end_y',
    20: 'nose_tip_x',
    21: 'nose_tip_y',
    22: 'mouth_left_corner_x',
    23: 'mouth_left_corner_y',
    24: 'mouth_right_corner_x',
    25: 'mouth_right_corner_y',
    26: 'mouth_center_top_lip_x',
    27: 'mouth_center_top_lip_y',
    28: 'mouth_center_bottom_lip_x',
    29: 'mouth_center_bottom_lip_y',
}
