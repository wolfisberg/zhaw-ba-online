import mir_eval
import numpy as np
import pysptk
import tensorflow as tf

import config


def convert_hz_to_cent(f, fref=10.0):
    return mir_eval.melody.hz2cents(f, fref)


def pitch_estimation_ref_rapt(x, fs, hop_size=256):
    f0 = pysptk.sptk.rapt(x/np.max(np.abs(x))*(2**15-1), fs, hop_size)
    return convert_hz_to_cent(f0)


def mix_noisy_speech(speech, noise):
    # todo: what is euclidean norm?
    speech_pow = tf.math.reduce_euclidean_norm(speech)
    noise_pow = tf.math.reduce_euclidean_norm(noise)

    min_SNR = config.SNR_RANGE[0]
    max_SNR = config.SNR_RANGE[1]

    # todo: why this formula and those magic numbers?
    snr_current = 20.0 * tf.math.log(speech_pow / noise_pow) / tf.math.log(10.0)
    snr_target = tf.random.uniform((), minval=min_SNR, maxval=max_SNR)

    # todo: why this formula and those magic numbers?
    noise = noise * tf.math.pow(10.0, (snr_current - snr_target) / 20.0)
    noisy_speech = speech + noise

    return speech, noise, noisy_speech


def get_random_slices(speech, noise):
    duration = 2  # seconds
    padding_end = 1  # seconds
    padding_start = 2  # seconds

    max_val = min(len(speech), len(noise)) - (duration + padding_end) * config.FS
    min_val = min(padding_start * config.FS, max_val - duration * config.FS)
    duration = min(duration, (min(len(speech), len(noise)) - max_val) / config.FS)
    random_start_idx_speech = int(tf.round(tf.random.uniform([], minval=min_val, maxval=max_val)))
    speech = speech[random_start_idx_speech:random_start_idx_speech + duration * config.FS]
    random_start_idx_noise = int(tf.round(tf.random.uniform([], minval=min_val, maxval=max_val)))
    noise = noise[random_start_idx_noise:random_start_idx_noise + duration * config.FS]
    return speech, noise, random_start_idx_speech, duration
