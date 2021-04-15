import tensorflow as tf
import numpy as np
from scipy import interpolate
import os

import config
import util


def _parse_noise_record(serialized_example):
    parsed_features = tf.io.parse_single_example(serialized_example, features=config.PARSING_CONFIG_NOISE)
    decoded_features = {
        "data_num_channels": tf.cast(parsed_features["data_num_channels"].values[0], tf.int32),
        "data_sampling_rate": tf.cast(parsed_features["data_sampling_rate"].values[0], tf.int32),
        "data_width": tf.cast(parsed_features["data_width"].values[0], tf.int32),
    }
    data = tf.io.decode_raw(parsed_features['data'].values[0], tf.int16)
    decoded_features.update({"data": data})
    return decoded_features


def _parse_speech_record(serialized_example):
    parsed_features = tf.io.parse_single_example(serialized_example, features=config.PARSING_CONFIG_SPEECH)
    decoded_features = {
        "data_num_channels": tf.cast(parsed_features["data_num_channels"].values[0], tf.int32),
        "data_sampling_rate": tf.cast(parsed_features["data_sampling_rate"].values[0], tf.int32),
        "data_width": tf.cast(parsed_features["data_width"].values[0], tf.int32),
        "pitch": tf.cast(parsed_features['pitch'].values, tf.float32),
        "pitch_confidence": tf.cast(parsed_features['pitch_confidence'].values, tf.float32),
    }
    data = tf.io.decode_raw(parsed_features['data'].values[0], tf.int16)
    decoded_features.update({"data": data})
    return decoded_features


def _interpolate_pitch(pitch, t):
    t_pitch = np.arange(0, len(pitch)) * config.PITCH_SAMPLING_TIME + config.PITCH_FRAME_LENGTH / 2
    f = interpolate.interp1d(t_pitch, pitch, 'nearest')
    return f(t).astype(np.float32)


def _calc_features(zipped):
    speech_data = zipped[0]
    noise_data = zipped[1]

    # Normalize values to [0,1]
    speech = tf.cast(speech_data["data"], tf.float32) / tf.int16.max
    noise = tf.cast(noise_data["data"], tf.float32) / tf.int16.max

    # Get X second slice from data
    speech, noise, random_start_idx, duration = util.get_random_slices(speech, noise)

    # Mix noise and speech with random SNR
    speech, noise, noisy = util.mix_noisy_speech(speech, noise)

    # Add gain
    random_gain = tf.math.exp(
        tf.random.uniform([], minval=tf.math.log(config.MIN_RAND_GAIN), maxval=tf.math.log(config.MAX_RAND_GAIN)))
    noisy = random_gain * noisy

    # todo: magic numbers
    interpolation_steps = np.array(
        range(random_start_idx, random_start_idx + duration * config.FS, config.FRAME_STEP)) / config.FS

    pitch = speech_data["pitch"]

    # Reduce low confidence pitches estimates to 0
    # pitch_confidence = speech_data["pitch_confidence"]
    # pitch = tf.where(pitch_confidence > config.pitch_confidence_threshold, pitch, 0)

    pitch_interpolated = _interpolate_pitch(pitch, interpolation_steps)
    pitch_interpolated_cents = util.convert_hz_to_cent(pitch_interpolated)

    return noisy.numpy(), pitch_interpolated_cents


def get_test_data():
    speech_ds = tf.data.TFRecordDataset(
        [os.path.join(config.SPEECH_DATA_TT_DIR, file) for file in os.listdir(config.SPEECH_DATA_TT_DIR)])
    speech_ds = speech_ds.map(_parse_speech_record)
    noise_ds = tf.data.TFRecordDataset(
        [os.path.join(config.NOISE_DATA_TT_DIR, file) for file in os.listdir(config.NOISE_DATA_TT_DIR)])
    noise_ds = noise_ds.map(_parse_noise_record)
    return list(map(_calc_features, zip(speech_ds, noise_ds)))
