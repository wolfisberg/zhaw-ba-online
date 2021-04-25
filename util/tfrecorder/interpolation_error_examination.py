import tensorflow as tf
import numpy as np
import librosa
import os
import pathlib
import matplotlib.pyplot as plt
import scipy.interpolate

_BASE_PATH = os.path.join('/' 'home', 'kaspar', 'Glacier', 'data-zhaw-ba', 'unzipped', 'MIR-1K')
_VALIDATION_DATA_DIR = 'cv'
_TRAINING_DATA_DIR = 'tr'
_TEST_DATA_DIR = 'tt'
_SHARD_BASE_NAME = 'shard_mir1k_'

# General
_MILLISECONDS_PER_SECOND = 1000

# Audio samples per seconds
_SAMPLE_RATE = 16000
_SAMPLE_RATE_MS = _SAMPLE_RATE / _MILLISECONDS_PER_SECOND

# CUT: section that is used to generate the frames
_CUT_LENGTH = 3

# Pitch estimations
_PITCH_LENGTH = 0.04
_PITCH_STEP = 0.02

# FRAME: unit which is fed to the NN, in number of samples
_FRAME_LENGTH = 1024
_FRAME_STEP = 256
_FRAME_COUNT = (_SAMPLE_RATE * _CUT_LENGTH) // _FRAME_STEP

length_differences = []


def main():
    create_tfrecords_from_directory(_BASE_PATH)


def create_tfrecords_from_directory(dir):
    for data_dir in [_TRAINING_DATA_DIR, _TEST_DATA_DIR, _VALIDATION_DATA_DIR]:
        data_dir_path = os.path.join(dir, data_dir)

        wav_files = librosa.util.find_files(directory=data_dir_path, ext=['wav'], recurse=False, case_sensitive=False)

        for file_path in wav_files:
            speech, sr = librosa.load(file_path, sr=_SAMPLE_RATE, mono=False)
            # _plot_stereo_audio(y, sr)
            speech = speech[1]
            # _plot_mono_audio(y, sr)

            if speech.dtype != np.dtype(np.int16):
                speech = (speech * np.iinfo(np.int16).max).astype(np.int16)

            file_name_stem = os.path.splitext(os.path.basename(file_path))[0]
            ref_matches = list(pathlib.Path(_BASE_PATH).rglob(f'*{file_name_stem}*.[pP][vV]'))
            if len(ref_matches) != 1:
                print(f'Cannot find ref (.pv) file for [ {file_path} ], skipping...')
                continue

            ref_pitch = np.genfromtxt(ref_matches[0])
            ref_pitch = _convert_semitone_to_hz(ref_pitch, 10)  # Convert semitones to Hz
            ref_pitch[ref_pitch <= 10] = 0  # Floor ref data 10 -> 0 Hz for model fitting
            ref_pitch = np.insert(ref_pitch, 0, 0)

            length_differences.append(
                _MILLISECONDS_PER_SECOND * len(speech) / _SAMPLE_RATE - len(ref_pitch) * _PITCH_STEP)

            random_start_idx = int(tf.round(tf.random.uniform([], minval=0, maxval=(
                    tf.cast(len(speech), tf.float32) - _CUT_LENGTH * _SAMPLE_RATE))))

            # Todo: try entire dataset with min and max values instead of random

            cut = speech[random_start_idx:random_start_idx + _CUT_LENGTH * _SAMPLE_RATE]

            frame_times_base = np.arange(0, _FRAME_COUNT)
            frame_times_base = frame_times_base * _FRAME_STEP / _SAMPLE_RATE
            start_index_offset = random_start_idx / _SAMPLE_RATE
            # Todo: can the first frame offset be ignored?
            # first_frame_offset = _FRAME_LENGTH / _SAMPLE_RATE
            # frame_times = start_index_offset + first_frame_offset + frame_times_base
            frame_times = start_index_offset + frame_times_base

            pitch_times_base = np.arange(0, len(ref_pitch))
            pitch_times = pitch_times_base * _PITCH_STEP

            frame_times[frame_times > np.max(pitch_times)] = np.max(pitch_times)
            frame_times[frame_times < np.min(pitch_times)] = np.min(pitch_times)

            if (np.max(pitch_times) < np.max(frame_times)):
                file_length = len(speech) / _SAMPLE_RATE
                pitch_max = np.max(pitch_times)
                frame_max = np.max(frame_times)
                print('above')

            if (np.min(pitch_times) > np.min(frame_times)):
                file_length = len(speech) / _SAMPLE_RATE
                pitch_min = np.min(pitch_times)
                frame_min = np.min(frame_times)
                print('above')

            f = scipy.interpolate.interp1d(pitch_times, ref_pitch, 'nearest')
            pitch_interpolated = f(frame_times).astype(np.float32)

            fig, ax = plt.subplots(sharex='all', sharey='all')
            spec, freqs, times, colormap = plt.specgram(speech, Fs=_SAMPLE_RATE)
            # plt.xlim(left=0.75, right=2)
            plt.ylim([0, 500])
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            fig.colorbar(colormap).set_label('Intensity [dB]')
            plt.plot(frame_times, pitch_interpolated, 'red')
            plt.plot(pitch_times, ref_pitch, 'orange')
            plt.show()

            print('done')

    print('done')


def _convert_semitone_to_hz(c, fref=10.0):
    return fref * 2 ** (c / 12.0)


if __name__ == '__main__':
    main()
