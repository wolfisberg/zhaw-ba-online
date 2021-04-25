import tensorflow as tf
import numpy as np
import librosa
from librosa import display
import os
import math
import pathlib
import matplotlib.pyplot as plt

_FILES_PER_SHARD = 750
_BASE_PATH = os.path.join('/' 'home', 'kaspar', 'Glacier', 'data-zhaw-ba', 'unzipped', 'MIR-1K')
_VALIDATION_DATA_DIR = 'cv'
_TRAINING_DATA_DIR = 'tr'
_TEST_DATA_DIR = 'tt'
_SHARD_BASE_NAME = 'shard_mir1k_'
_SAMPLE_RATE = 16000

length_differences = []

def main():
    create_tfrecords_from_directory(_BASE_PATH)


def create_tfrecords_from_directory(dir):
    for data_dir in [_TRAINING_DATA_DIR, _TEST_DATA_DIR, _VALIDATION_DATA_DIR]:
        data_dir_path = os.path.join(dir, data_dir)

        wav_files = librosa.util.find_files(directory=data_dir_path, ext=['wav'], recurse=False, case_sensitive=False)
        number_of_shards = math.ceil(len(wav_files) / _FILES_PER_SHARD)

        for shard_number in range(number_of_shards):
            shard_name = f'{_SHARD_BASE_NAME}_{data_dir}_{str(shard_number).rjust(4, "0")}.tfrecord'
            shard_path = os.path.join(_BASE_PATH, 'tfrecords', data_dir, shard_name)

            with tf.io.TFRecordWriter(shard_path) as out:
                lowerIndex = shard_number * _FILES_PER_SHARD
                upperIndex = (shard_number + 1) * _FILES_PER_SHARD

                for fileIndex in range(lowerIndex, upperIndex if upperIndex <= len(wav_files) else len(wav_files)):
                    file_path = wav_files[fileIndex]
                    y, sr = librosa.load(file_path, sr=_SAMPLE_RATE, mono=False)
                    # _plot_stereo_audio(y, sr)
                    y = y[1]
                    # _plot_mono_audio(y, sr)

                    if y.dtype != np.dtype(np.int16):
                        y = (y * np.iinfo(np.int16).max).astype(np.int16)

                    file_name_stem = os.path.splitext(os.path.basename(file_path))[0]
                    ref_matches = list(pathlib.Path(_BASE_PATH).rglob(f'*{file_name_stem}*.[pP][vV]'))
                    if len(ref_matches) != 1:
                        print(f'Cannot find ref (.pv) file for [ {file_path} ], skipping...')
                        continue

                    ref_data = np.genfromtxt(ref_matches[0])
                    ref_data = _convert_semitone_to_hz(ref_data, 10)  # Convert semitones to Hz
                    ref_data[ref_data <= 10] = 0  # Floor ref data 10 -> 0 Hz for model fitting
                    ref_data = np.insert(ref_data, 0, 0)
                    length_differences.append(len(y) / 16000 * 1000 - len(ref_data) * 20)

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'data': _bytes_feature(y.tobytes()),
                        'data_sampling_rate': _int64_feature([sr]),
                        'data_num_channels': _int64_feature([1]),
                        'data_width': _int64_feature([len(y)]),
                        'pitch': _float_feature(ref_data),
                        'pitch_confidence': _float_feature(np.ones(len(ref_data))),
                    }))

                    out.write(example.SerializeToString())

    print('done')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _convert_semitone_to_hz(c, fref=10.0):
    return fref * 2 ** (c / 12.0)



def _plot_stereo_audio(y, sr):
    fig, ax = plt.subplots(nrows=2, sharex='all', sharey='all')
    librosa.display.waveplot(y[0], sr=sr, ax=ax[0])
    ax[0].set(title='Channel 0')
    ax[0].label_outer()
    librosa.display.waveplot(y[1], sr=sr, ax=ax[1])
    ax[1].set(title='Channel 1')
    ax[1].label_outer()
    fig.show()

def _plot_mono_audio(y, sr):
    fig, ax = plt.subplots()
    librosa.display.waveplot(y, sr=sr)
    ax.set(title='Mono')
    ax.label_outer()
    fig.show()


main()
