import tensorflow as tf
import os
import datetime


# Audio
SNR_RANGE = (-5.0,20.0) #dB
FRAME_LENGTH = 512
FRAME_STEP = 256
MIN_RAND_GAIN = 0.05
MAX_RAND_GAIN = 1.1
SAMPLE_LENGTH = 3 #shorter than shortest noise/speech sample
FS = 16000
PITCH_SAMPLING_TIME = 0.01 # s
PITCH_FRAME_LENGTH = 0.032 # s


# Data
BATCH_SIZE = 64
NUM_FRAMES = 1 + FS * SAMPLE_LENGTH - FRAME_LENGTH // FRAME_STEP


# Training
STEPS_PER_EPOCH = 720
EPOCHS = 100
VALIDATION_STEPS = 70


# Directories
_DATA_DIR = os.path.join('..', 'data', 'tfrecords')
_TFRECORDS_DIR = os.path.join(_DATA_DIR, 'tfrecords')

SPEECH_DATA_TR_DIR = os.path.join(_TFRECORDS_DIR, 'speech', 'tr')
NOISE_DATA_TR_DIR = os.path.join(_TFRECORDS_DIR, 'noise', 'tr')
SPEECH_DATA_CV_DIR = os.path.join(_TFRECORDS_DIR, 'speech', 'cv')
NOISE_DATA_CV_DIR = os.path.join(_TFRECORDS_DIR, 'noise', 'cv')
SPEECH_DATA_TT_DIR = os.path.join(_TFRECORDS_DIR, 'speech', 'tt')
NOISE_DATA_TT_DIR = os.path.join(_TFRECORDS_DIR, 'noise', 'tt')

_date_time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = os.path.join(_DATA_DIR, 'logs', _date_time_string)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
CHECKPOINT_DIR = os.path.join(_DATA_DIR, 'checkpoints', _date_time_string)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)


# Misc
SEED = 2


# Parsing
PARSING_CONFIG_NOISE = {
    'data': tf.io.VarLenFeature(tf.string),
    'data_sampling_rate': tf.io.VarLenFeature(tf.int64),
    'data_num_channels': tf.io.VarLenFeature(tf.int64),
    'data_width': tf.io.VarLenFeature(tf.int64),
}

PARSING_CONFIG_SPEECH = {
    'data': tf.io.VarLenFeature(tf.string),
    'data_sampling_rate': tf.io.VarLenFeature(tf.int64),
    'data_num_channels': tf.io.VarLenFeature(tf.int64),
    'data_width': tf.io.VarLenFeature(tf.int64),
    'pitch': tf.io.VarLenFeature(tf.float32),
    'pitch_confidence': tf.io.VarLenFeature(tf.float32),
}

