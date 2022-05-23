"""
This code is based on https://github.com/mingzhaochina/unet_cea
"""

import os
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import csv
import json
from obspy.core.utcdatetime import UTCDateTime

tf.compat.v1.disable_eager_execution()

class DataWriter(object):
    def __init__(self, filename):
        self._writer = None
        self._filename = filename
        self._written = 0
        self._writer = tf.io.TFRecordWriter(self._filename)

    def write(self, sample_window, labels):
        n_traces = len(sample_window)
        n_samples = len(sample_window[0].data)

        # Extract data
        data = np.zeros((n_traces, n_samples), dtype=np.float32)
        for i in range(n_traces):
            data[i, :] = sample_window[i].data[...]

        #used for one-hot label
        label = np.zeros((n_traces, n_samples), dtype=np.float32)
        for i in range(n_traces):
            label[i, :] = labels[i].data[...]
        

        # Extract metadata (csv file)
        start_time = np.int64(sample_window[0].stats.starttime.timestamp)
        end_time = np.int64(sample_window[0].stats.endtime.timestamp)

        example = tf.train.Example(features=tf.train.Features(feature={
            'trace_name': self._bytes_feature(trace_name.encode('utf-8')),
            'window_size': self._int64_feature(n_samples),
            'n_traces': self._int64_feature(n_traces),
            'data': self._bytes_feature(data.tobytes()),
            'label': self._bytes_feature(label.tobytes()),
            'start_time': self._int64_feature(start_time),
            'end_time': self._int64_feature(end_time),
        }))
        self._writer.write(example.SerializeToString())
        self._written += 1

    def close(self):
        self._writer.close()

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(
                                    value=value.flatten().tolist()))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# reading data from stream
class DataReader(object): 
    def __init__(self, path, config, shuffle=True):
        self._path = path
        self._shuffle = shuffle
        self._config = config
        self.win_size = config.win_size
        self.n_traces = config.n_traces
        self._reader = tf.compat.v1.TFRecordReader()

    def read(self):
        filename_queue = self._filename_queue()
        _, serialized_example = self._reader.read(filename_queue)
        example = self._parse_example(serialized_example)
        return example

    def _filename_queue(self):
        fnames = []
        for root, dirs, files in os.walk(self._path):
            for f in files:
                if f.endswith(".tfrecords"):
                    fnames.append(os.path.join(root, f))
        fname_q = tf.compat.v1.train.string_input_producer(fnames, shuffle=self._shuffle, num_epochs = self._config.n_epochs)
        return fname_q

    def _parse_example(self, serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'trace_name': tf.io.FixedLenFeature([], tf.string),
                'window_size': tf.io.FixedLenFeature([], tf.int64),
                'n_traces': tf.io.FixedLenFeature([], tf.int64),
                'data': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.string),
                'start_time': tf.io.FixedLenFeature([],tf.int64),
                'end_time': tf.io.FixedLenFeature([], tf.int64)})

        # Convert and reshape
        data = tf.io.decode_raw(features['data'], tf.float32)
        # print ("data",data)
        data.set_shape([self.n_traces * self.win_size])
        data = tf.reshape(data, [self.n_traces, self.win_size])
        data = tf.transpose(data, [1, 0])
        # Pack
        features['data'] = data

        label = tf.io.decode_raw(features['label'], tf.float32)
        label.set_shape([self.n_traces * self.win_size])
        label = tf.reshape(label, [self.n_traces, self.win_size])
        label = tf.transpose(label, [1, 0])
        features['label'] = label
        return features


class DataPipeline(object):
    """Creates a queue op to stream data for training.
    Attributes:
    samples: Tensor(float). batch of input samples [batch_size, n_channels, n_points]
    labels: Tensor(int32). Corresponding batch labels, [batch_size, n_channels, n_points]
    """

    def __init__(self, dataset_path, config, is_training):

        min_after_dequeue = 1000
        capacity = 1000 + 3 * config.batch_size
        if is_training:
            with tf.name_scope('inputs'):
                self._reader = DataReader(dataset_path, config=config)
                samples = self._reader.read()

                sample_input = samples["data"]
                sample_target = samples["label"]
                start_time = samples["start_time"]
                end_time = samples["end_time"]

                self.samples, self.labels, self.start_time, self.end_time = tf.compat.v1.train.batch(
                    [sample_input, sample_target, start_time, end_time],
                    batch_size = config.batch_size,
                    capacity = capacity,
                    num_threads = config.n_threads,
                    allow_smaller_final_batch = False)

        elif not is_training:

            with tf.name_scope('validation_inputs'):
                self._reader = DataReader(dataset_path, config=config)
                samples = self._reader.read()
                sample_trace_name = samples['trace_name']
                sample_input = samples["data"]
                sample_target = samples["label"]
                start_time = samples["start_time"]
                end_time = samples["end_time"]

                self.traces, self.samples, self.labels, self.start_time, self.end_time = tf.compat.v1.train.batch(
                    [sample_trace_name, sample_input, sample_target, start_time, end_time],
                    batch_size = config.batch_size,
                    capacity = capacity,
                    num_threads = config.n_threads,
                    allow_smaller_final_batch = False)
        else:
            raise ValueError(
                "is_training flag is not defined, set True for training and False for testing")

