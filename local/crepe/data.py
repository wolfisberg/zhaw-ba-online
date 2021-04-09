import tensorflow as tf
import os
import config
import numpy as np
import scipy.interpolate


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


def _mix_noisy_speech(speech, noise):
    speech_pow = tf.math.reduce_euclidean_norm(speech)
    noise_pow = tf.math.reduce_euclidean_norm(noise)

    min_SNR = config.SNR_RANGE[0]
    max_SNR = config.SNR_RANGE[1]
    snr_current = 20.0*tf.math.log(speech_pow/noise_pow)/tf.math.log(10.0)
    snr_target = tf.random.uniform((),minval=min_SNR,maxval=max_SNR)

    noise = noise * tf.math.pow(10.0,(snr_current-snr_target)/20.0)
    noisy_speech = speech+noise

    return speech, noise, noisy_speech


def _interpolate_pitch(pitch,t):
    pitch = pitch.numpy()
    t = t.numpy()
    t_pitch = np.arange(0, len(pitch)) * config.PITCH_SAMPLING_TIME + config.PITCH_FRAME_LENGTH / 2
    f = scipy.interpolate.interp1d(t_pitch, pitch, 'nearest')
    return f(t).astype(np.float32)


#@tf.function
def _interpolate_pitch_tf(pitch,t):
    y = tf.py_function(_interpolate_pitch,[pitch,t], Tout=tf.float32)
    return tf.squeeze(y)


def _calc_features(speech_data, noise_data):
    speech = tf.squeeze(tf.cast(speech_data["data"], tf.float32))
    noise = tf.squeeze(tf.cast(noise_data["data"], tf.float32))
    speech = speech / tf.int16.max
    noise = noise / tf.int16.max

    random_start_idx = int(tf.round(tf.random.uniform([], maxval=(
            tf.cast(len(noise), tf.float32) - config.SAMPLE_LENGTH * config.FS - config.PITCH_SAMPLING_TIME))))
    noise = noise[random_start_idx:random_start_idx + config.SAMPLE_LENGTH * config.FS]

    random_start_idx = int(tf.round(tf.random.uniform([], minval=161, maxval=(
            tf.cast(len(speech), tf.float32) - config.SAMPLE_LENGTH * config.FS - 161))))
    speech = speech[random_start_idx:random_start_idx + config.SAMPLE_LENGTH * config.FS]

    #SNR_range = config.SNR_RANGE
    frame_length = config.FRAME_LENGTH
    frame_step = config.FRAME_STEP
    speech, noise, noisy = _mix_noisy_speech(speech, noise)

    random_gain = tf.math.exp(
        tf.random.uniform([], minval=tf.math.log(config.MIN_RAND_GAIN), maxval=tf.math.log(config.MAX_RAND_GAIN)))
    noisy = random_gain * noisy

    noisy_frames = tf.signal.frame(noisy, frame_length, frame_step)
    #noisy_stft = tf.signal.stft(noisy,frame_length,frame_step)
    frame_times = random_start_idx / config.FS + tf.range(0, config.NUM_FRAMES) * frame_step / config.FS + frame_length / config.FS

    pitch = tf.squeeze(speech_data["pitch"])
    #pitch_confidence = tf.squeeze(speech_data["pitch_confidence"])
    #pitch = tf.where(pitch_confidence>config['pitch_confidence_threshold'],pitch,0)
    pitch_interpolated = _interpolate_pitch_tf(pitch, frame_times)
    return noisy_frames, pitch_interpolated


def get_training_data():
    speech_ds = tf.data.TFRecordDataset([os.path.join(config.SPEECH_DATA_TR_DIR, file) for file in os.listdir(config.SPEECH_DATA_TR_DIR)])
    speech_ds = speech_ds.map(_parse_speech_record).repeat(None).shuffle(buffer_size=1000, seed=config.SEED)

    noise_ds = tf.data.TFRecordDataset([os.path.join(config.NOISE_DATA_TR_DIR, file) for file in os.listdir(config.NOISE_DATA_TR_DIR)])
    noise_ds = noise_ds.map(_parse_noise_record).repeat(None).shuffle(buffer_size=1000, seed=config.SEED)

    dataset_combined = tf.data.Dataset.zip((speech_ds, noise_ds))
    dataset_features = dataset_combined.map(_calc_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_features = dataset_features.batch(config.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset_features


def get_validation_data():
    speech_ds = tf.data.TFRecordDataset([os.path.join(config.SPEECH_DATA_CV_DIR, file) for file in os.listdir(config.SPEECH_DATA_CV_DIR)])
    speech_ds = speech_ds.map(_parse_speech_record).repeat(None).shuffle(buffer_size=1000, seed=config.SEED)

    noise_ds = tf.data.TFRecordDataset([os.path.join(config.NOISE_DATA_CV_DIR, file) for file in os.listdir(config.NOISE_DATA_CV_DIR)])
    noise_ds = noise_ds.map(_parse_noise_record).repeat(None).shuffle(buffer_size=1000, seed=config.SEED)

    dataset_combined = tf.data.Dataset.zip((speech_ds, noise_ds))
    dataset_features = dataset_combined.map(_calc_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_features = dataset_features.batch(config.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset_features


def get_test_data():
    speech_ds = tf.data.TFRecordDataset([os.path.join(config.SPEECH_DATA_TT_DIR, file) for file in os.listdir(config.SPEECH_DATA_TT_DIR)])
    speech_ds = speech_ds.map(_parse_speech_record).repeat(None).shuffle(buffer_size=1000, seed=config.SEED)

    noise_ds = tf.data.TFRecordDataset([os.path.join(config.NOISE_DATA_TT_DIR, file) for file in os.listdir(config.NOISE_DATA_TT_DIR)])
    noise_ds = noise_ds.map(_parse_noise_record).repeat(None).shuffle(buffer_size=1000, seed=config.SEED)

    dataset_combined = tf.data.Dataset.zip((speech_ds, noise_ds))
    dataset_features = dataset_combined.map(_calc_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_features = dataset_features.batch(config.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset_features
