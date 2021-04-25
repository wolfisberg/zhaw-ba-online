import os
import sys
import librosa
import soundfile
from datetime import datetime
import tensorflow as tf

SR_TARGET = 16000
MIXED_DIR = os.path.join('data', 'mixed')
if not os.path.exists(MIXED_DIR):
    os.makedirs(MIXED_DIR)


def parse_args():
    speech_file = os.path.join('data', 'speech', 'mic_F04_si1021.wav')
    noise_file = os.path.join('data', 'noise', '01aa010e_2.4468_20ea0105_-2.4468.wav')
    snr_targets = list(range(-10, 26, 5))
    if len(sys.argv) > 3:
        speech_file = sys.argv[1]
        noise_file = sys.argv[2]
        snr_targets = [int(i) for i in sys.argv[3:]]
    return speech_file, noise_file, snr_targets


def mix_noise_speech(speech_path, noise_path, snr_target):
    # Load wav files
    speech, sr_speech = librosa.load(path=speech_path, sr=SR_TARGET, mono=True)
    noise, sr_noise = librosa.load(path=noise_path, sr=SR_TARGET, mono=True)

    # Mix speech with noise
    speech_pow = tf.math.reduce_euclidean_norm(speech)
    noise_pow = tf.math.reduce_euclidean_norm(noise)
    snr_current = 20.0 * tf.math.log(speech_pow / noise_pow) / tf.math.log(10.0)
    noise_snr_adjusted = noise * tf.math.pow(10.0, (snr_current - snr_target) / 20.0)
    length = min(len(speech), len(noise_snr_adjusted))
    noisy_speech = speech[-length:] + noise_snr_adjusted[-length:]

    # Write out mixed audio to file
    out_path = os.path.join(MIXED_DIR, f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_mixed_snr_{snr_target}.wav')
    print(f'Mixed file written to {out_path}')
    soundfile.write(
        file=out_path,
        data=noisy_speech,
        samplerate=SR_TARGET,
        format='WAV',
        subtype='PCM_16')


def main():
    speech_file, noise_file, snr_targets = parse_args()
    for snr in snr_targets:
        mix_noise_speech(speech_file, noise_file, snr)


if __name__ == '__main__':
    main()
