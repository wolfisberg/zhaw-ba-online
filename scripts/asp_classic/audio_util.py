import numpy as np
import tensorflow as tf
from scipy import interpolate

from ..asp_classic import config


def interpolate_pitch(pitch, t):
    t_pitch = (np.arange(0, len(pitch)) * config.PITCH_SAMPLING_TIME + config.PITCH_FRAME_LENGTH / 2)
    f = interpolate.interp1d(t_pitch, pitch, "nearest")
    return f(t).astype(np.float32)


def mix_noisy_speech(speech, noise):
    # todo: what is euclidean norm?
    # todo: maybe has to be squared?
    # Leistung des Signals
    speech_pow = tf.math.reduce_euclidean_norm(speech)
    noise_pow = tf.math.reduce_euclidean_norm(noise)

    snr_current = 20.0 * tf.math.log(speech_pow / noise_pow) / tf.math.log(10.0)
    snr_target = tf.random.uniform((), minval=config.SNR_RANGE[0], maxval=config.SNR_RANGE[1])
    noise = noise * tf.math.pow(10.0, (snr_current - snr_target) / 20.0)
    noisy_speech = speech + noise

    return noisy_speech


def get_random_slice(speech, noise):
    duration = 2  # seconds
    padding_end = 1  # seconds
    padding_start = 2  # seconds

    max_val = min(len(speech), len(noise)) - (duration + padding_end) * config.FS
    min_val = min(padding_start * config.FS, max_val - duration * config.FS)
    duration = min(duration, (min(len(speech), len(noise)) - max_val) / config.FS)
    random_start_idx_speech = int(
        tf.round(tf.random.uniform([], minval=min_val, maxval=max_val))
    )
    speech = speech[
        random_start_idx_speech : random_start_idx_speech + duration * config.FS
    ]
    random_start_idx_noise = int(
        tf.round(tf.random.uniform([], minval=min_val, maxval=max_val))
    )
    noise = noise[random_start_idx_noise : random_start_idx_noise + duration * config.FS]

    if random_start_idx_speech < 0:
        print("This should not happen.")

    return speech, noise, random_start_idx_speech, duration


def get_interpolationsteps_for_stepsize(start_idx, duration, step_size):
    return (np.array(range(start_idx, start_idx + duration * config.FS, step_size)) / config.FS)
