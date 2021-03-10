import tensorflow as tf
import numpy as np
import librosa
import os
import math
import scipy
import pathlib


_FILES_PER_SHARD = 750
_SPEECH_DATA_DIR = 'speech'
_NOISE_DATA_DIR = 'noise'
_VALIDATION_DATA_DIR = 'cv'
_TRAINING_DATA_DIR = 'tr'
_TEST_DATA_DIR = 'tt'
_SHARD_BASE_NAME = 'shard'
_SAMPLE_RATE = 16000


def main():
    create_tfrecords_from_directory(_SPEECH_DATA_DIR, add_ref_data=True)
    create_tfrecords_from_directory(_NOISE_DATA_DIR, add_ref_data=False)


def create_tfrecords_from_directory(dir, add_ref_data=False):
    for data_dir in [_TRAINING_DATA_DIR, _TEST_DATA_DIR, _VALIDATION_DATA_DIR]:
        data_dir_path = os.path.join('data', 'downsampled', dir, data_dir)
        wav_files = librosa.util.find_files(directory=data_dir_path, ext=['wav'], recurse=False, case_sensitive=False)
        number_of_shards = math.ceil(len(wav_files) / _FILES_PER_SHARD)

        for shard_number in range(number_of_shards):
            shard_name = f'{_SHARD_BASE_NAME}_{dir}_{data_dir}_{str(shard_number).rjust(4, "0")}.tfrecord'
            shard_path = os.path.join('data', 'tfrecords', dir, shard_name)

            with tf.io.TFRecordWriter(shard_path) as out:
                lowerIndex = shard_number * _FILES_PER_SHARD
                upperIndex = (shard_number + 1) * _FILES_PER_SHARD

                for fileIndex in range(lowerIndex, upperIndex if upperIndex <= len(wav_files) else len(wav_files)):
                    file_path = wav_files[fileIndex]
                    # read via librosa
                    y, sr = librosa.load(file_path, sr=None)

                    # read via tensorflow
                    # raw_audio = tf.io.read_file(file_path)
                    # y, sr = tf.audio.decode_wav(raw_audio)

                    # read via scipy
                    # sr, y = scipy.io.wavfile.read(file_path)

                    if sr != _SAMPLE_RATE:
                        y = librosa.to_mono(y)
                        y = librosa.resample(y=y, orig_sr=sr, target_sr=_SAMPLE_RATE, res_type='kaiser_best', scale=True)

                    if y.dtype != np.dtype(np.int16):
                        y = (y / np.max(np.abs(y)) * np.iinfo(np.int16).max).astype(np.int16)

                    if add_ref_data:
                        file_name_stem = os.path.splitext(os.path.basename(file_path))[0][4:]
                        ref_matches = list(pathlib.Path(data_dir_path).rglob(f'*{file_name_stem}*.[fF]0'))
                        if len(ref_matches) != 1:
                            print(f'Cannot find ref (f0) file for [ {file_path} ], skipping...')
                            continue

                        ref_data = np.genfromtxt(ref_matches[0], delimiter=' ')

                        example = tf.train.Example(features=tf.train.Features(feature={
                            'data': _bytes_feature(y.tobytes()),
                            'data_sampling_rate': _int64_feature([sr]),
                            'data_num_channels': _int64_feature([1]),
                            'data_width': _int64_feature([len(y)]),
                            'pitch': _float_feature(ref_data.T[2]),
                            'pitch_confidence': _float_feature(ref_data.T[3]),
                        }))

                    else:
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'data': _bytes_feature(y.tobytes()),
                            'data_sampling_rate': _int64_feature([sr]),
                            'data_num_channels': _int64_feature([1]),
                            'data_width': _int64_feature([len(y)]),
                        }))


                    out.write(example.SerializeToString())


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


main()
