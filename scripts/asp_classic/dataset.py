import tensorflow as tf
import os

from ..asp_classic import config, audio_util


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


def _calc_features(zipped, mix_with_noise):
    speech_data = zipped[0]
    noise_data = zipped[1]

    # Normalize values to [0,1]
    speech = tf.cast(speech_data["data"], tf.float32) / tf.int16.max
    noise = tf.cast(noise_data["data"], tf.float32) / tf.int16.max

    if len(speech) == 0:
        print('this should not happen')

    if len(speech) == 0:
        print('this should not happen')

    # Get X second slice from data
    speech, noise, random_start_idx, duration = audio_util.get_random_slice(speech, noise)

    # Mix noise and speech with random SNR
    if mix_with_noise:
        speech = audio_util.mix_noisy_speech(speech, noise)

    # Add gain
    random_gain = tf.math.exp(
        tf.random.uniform([], minval=tf.math.log(config.MIN_RAND_GAIN), maxval=tf.math.log(config.MAX_RAND_GAIN)))
    speech = random_gain * speech

    interpolation_steps = [audio_util.get_interpolationsteps_for_stepsize(random_start_idx, duration, s)
                           for s in config.FRAME_STEPS]

    # Remove the padding added to the tfrecords
    pitch = speech_data["pitch"][6:]

    # Reduce low confidence pitches estimates to 0
    # pitch_confidence = speech_data["pitch_confidence"]
    # pitch = tf.where(pitch_confidence > config.pitch_confidence_threshold, pitch, 0)

    pitches_interpolated = [audio_util.interpolate_pitch(pitch, s) for s in interpolation_steps]

    return speech.numpy(), *pitches_interpolated


def get_test_data(mix_with_noise=True):
    speech_ds = tf.data.TFRecordDataset(
        [os.path.join(config.SPEECH_DATA_TT_DIR, file) for file in os.listdir(config.SPEECH_DATA_TT_DIR)])
    speech_ds = speech_ds.map(_parse_speech_record)
    noise_ds = tf.data.TFRecordDataset(
        [os.path.join(config.NOISE_DATA_TT_DIR, file) for file in os.listdir(config.NOISE_DATA_TT_DIR)])
    noise_ds = noise_ds.map(_parse_noise_record)
    zipped = zip(speech_ds, noise_ds)
    return [_calc_features(z, mix_with_noise) for z in zipped]
