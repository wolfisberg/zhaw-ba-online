import tensorflow as tf
import os


record_file = os.path.join('data', 'tfrecords', 'speech', 'shard_speech_tr_0000.tfrecord')
raw_dataset = tf.data.TFRecordDataset(record_file)


# Format used for writing tfrecord:
# example = tf.train.Example(features=tf.train.Features(feature={
#     'data': _bytes_feature(y.tobytes()),
#     'data_sampling_rate': _int64_feature([sr]),
#     'data_num_channels': _int64_feature([1]),
#     'data_width': _int64_feature([len(y)]),
#     'pitch': _float_feature(ref_data.T[2]),
#     'pitch_confidence': _float_feature(ref_data.T[3]),
# }))


def parse_record(record):
    name_to_features = {
        'data': tf.io.VarLenFeature(tf.string),
        'data_sampling_rate': tf.io.VarLenFeature(tf.int64),
        'data_num_channels': tf.io.VarLenFeature(tf.int64),
        'data_width': tf.io.VarLenFeature(tf.int64),
        'pitch': tf.io.VarLenFeature(tf.float32),
        'pitch_confidence': tf.io.VarLenFeature(tf.float32),
    }
    return tf.io.parse_single_example(record, name_to_features)


def decode_record(record):
    wav_data = record['data'].values[0]
    sampling_rate = record['data_sampling_rate'].values[0]
    number_of_channels = record['data_num_channels'].values[0]
    data_width = record['data_width'].values[0]
    pitch = record['pitch']
    pitch_confidence = record['pitch_confidence']
    return (wav_data, sampling_rate, number_of_channels, data_width, pitch, pitch_confidence)


for record in raw_dataset:
    parsed_record = parse_record(record)
    wav_data, sampling_rate, number_of_channels, data_width, pitch, pitch_confidence = decode_record(parsed_record)

