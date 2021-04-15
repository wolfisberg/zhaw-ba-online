import os
import librosa
import soundfile
import numpy as np
from datetime import datetime
import tensorflow as tf


# Config
SR_TARGET = 16000
SNR_TARGET = 25
MIXED_DIR = os.path.join('data', 'mixed')
if not os.path.exists(MIXED_DIR):
    os.makedirs(MIXED_DIR)

# Load wav files
speech_file = os.path.join('data', 'speech', 'mic_F04_si1021.wav')
noise_file = os.path.join('data', 'noise', '01aa010e_2.4468_20ea0105_-2.4468.wav')
speech_raw, sr_speech = librosa.load(path=speech_file, sr=None)
noise_raw, sr_noise = librosa.load(path=noise_file, sr=None)

# Resample to TARGET_SNR
speech_resampled = librosa.resample(
    y=librosa.to_mono(speech_raw),
    orig_sr=sr_speech,
    target_sr=SR_TARGET,
    res_type='kaiser_best',
    scale=True)
noise_resampled = librosa.resample(
    y=librosa.to_mono(noise_raw),
    orig_sr=sr_noise,
    target_sr=SR_TARGET,
    res_type='kaiser_best',
    scale=True)

# Set bit depth to 16
speech_resampled_normalized =\
    (speech_resampled / np.max(np.abs(speech_resampled))* np.iinfo(np.int16).max).astype(np.int16)
noise_resampled_normalized = \
    (noise_resampled / np.max(np.abs(noise_resampled))* np.iinfo(np.int16).max).astype(np.int16)

# Mix speech with noise
speech_pow = tf.math.reduce_euclidean_norm(speech_resampled_normalized)
noise_pow = tf.math.reduce_euclidean_norm(noise_resampled_normalized)
snr_current = 20.0 * tf.math.log(speech_pow / noise_pow) / tf.math.log(10.0)
noise_snr_adjusted = noise_resampled_normalized * tf.math.pow(10.0, (snr_current - SNR_TARGET) / 20.0)
length = min(len(speech_resampled_normalized), len(noise_snr_adjusted))
#noisy_speech = speech_resampled_normalized[-length:] + noise_snr_adjusted[-length:]
noisy_speech = speech_resampled_normalized[-length:] + noise_resampled_normalized[-length:]

# Write out mixed audio to file
soundfile.write(
    file=os.path.join(MIXED_DIR, f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_mixed_snr_{SNR_TARGET}.wav'),
    data=noisy_speech,
    samplerate=SR_TARGET,
    format='WAV',
    subtype='PCM_16')
