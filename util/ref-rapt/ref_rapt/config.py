import os
import tensorflow as tf


# Audio
SNR_RANGE = (-5.0,20.0) #dB
FRAME_STEP = 256
MIN_RAND_GAIN = 0.05
MAX_RAND_GAIN = 1.1
FS = 16000
PITCH_SAMPLING_TIME = 0.01 # s
PITCH_FRAME_LENGTH = 0.032 # s


# Directories
_DATA_DIR = os.path.join('/home/kaspar/Glacier/data-zhaw-ba')
_TFRECORDS_DIR = os.path.join(_DATA_DIR, 'tfrecords')

SPEECH_DATA_TT_DIR = os.path.join(_TFRECORDS_DIR, 'speech', 'tt')
NOISE_DATA_TT_DIR = os.path.join(_TFRECORDS_DIR, 'noise', 'tt')


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
