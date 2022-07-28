#!/usr/bin/env python

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to create SSL splits from a dataset.
"""

import json
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import trange, tqdm

from libml import data as libml_data
from libml import utils

flags.DEFINE_integer('seed', 0, 'Random seed to use, 0 for no shuffling.')
flags.DEFINE_integer('size', 0, 'Size of labelled set.')
flags.DEFINE_string('labels', '', 'Numpy array for Dr')
flags.DEFINE_string('labeled', '', 'Labeled samples')
flags.DEFINE_string('name', '', 'Name')

FLAGS = flags.FLAGS


def get_class(serialized_example):
    return tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64)})['label']


def main(argv):
    assert FLAGS.size
    argv.pop(0)
    if any(not tf.gfile.Exists(f) for f in argv[1:]):
        raise FileNotFoundError(argv[1:])

    target = FLAGS.name
    target = os.path.join('data/SSL2', target)
    #target = '%s.%d@%d' % (argv[0], FLAGS.seed, FLAGS.size)
    if tf.gfile.Exists(target):
        raise FileExistsError('For safety overwriting is not allowed', target)
    input_files = argv[1:]
    count = 0
    id_class = []
    class_id = defaultdict(list)
    print('Computing class distribution')
    dataset = tf.data.TFRecordDataset(input_files).map(get_class, 4).batch(1 << 10)
    print('input_files', input_files)
    it = dataset.make_one_shot_iterator().get_next()
    try:
        with tf.Session() as session, tqdm(leave=False) as t:
            while 1:
                old_count = count
                for i in session.run(it):
                    id_class.append(i)
                    class_id[i].append(count)
                    count += 1
                t.update(count - old_count)
    except tf.errors.OutOfRangeError:
        pass
    print('%d records found' % count)
    nclass = len(class_id)
    assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
    train_stats = np.array([len(class_id[i]) for i in range(nclass)], np.float64)
    train_stats /= train_stats.max()
    if 'stl10' in argv[1]:
        # All of the unlabeled data is given label 0, but we know that
        # STL has equally distributed data among the 10 classes.
        train_stats[:] = 1

    print('  Stats', ' '.join(['%.2f' % (100 * x) for x in train_stats]))
    assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
    class_id = [np.array(class_id[i], dtype=np.int64) for i in range(nclass)]
    if FLAGS.seed:
        np.random.seed(FLAGS.seed)
        for i in range(nclass):
            np.random.shuffle(class_id[i])

    # Distribute labels to match the input distribution.
    '''
    npos = np.zeros(nclass, np.int64)
    label = []
    for i in range(FLAGS.size):
        c = np.argmax(train_stats - npos / max(npos.max(), 1))
        label.append(class_id[c][npos[c]])
        npos[c] += 1
    '''
    print('label', FLAGS.labels)
    print('relabel', FLAGS.labeled)
    # label = np.argmax(np.load(FLAGS.labels)['arr_0'], axis=1)
    # relabel = np.load(FLAGS.labeled)['arr_0']
    label = np.load(FLAGS.labeled)['arr_0']
    relabel = np.load(FLAGS.labels)['arr_0']
    print('label, relabel', label.shape, relabel.shape)
    del class_id
    label = frozenset([int(x) for x in label])
    if 'stl10' in argv[1] and FLAGS.size == 1000:
        data = tf.gfile.Open(os.path.join(libml_data.DATA_DIR, 'stl10_fold_indices.txt'), 'r').read()
        label = frozenset(list(map(int, data.split('\n')[FLAGS.seed].split())))

    print('Creating split in %s' % target)
    tf.gfile.MakeDirs(os.path.dirname(target))

    select_idx = np.load('../select_idx.npz')['arr_0']
    print('select_idx', select_idx.shape)
    ori_label = dict()
    for i, ii in enumerate(select_idx):
        ori_label[ii] = i

    with tf.python_io.TFRecordWriter(target + '-label.tfrecord') as writer_label:
        print('writer_label', target + '-label.tfrecord')
        pos, loop = 0, trange(count, desc='Writing records')
        a = 0
        total = 0
        N = np.zeros([nclass])
        for input_file in input_files:
            # print('input_file', input_file)
            for record in tf.python_io.tf_record_iterator(input_file):
                if pos in ori_label.keys() and ori_label[pos] in label:
                    #Modifing label with the guessed one
                    s = tf.train.Example.FromString(record)
                    f = dict(s.features.feature)
                    guessed_label = np.argmax(relabel[ori_label[pos]])
                    N[guessed_label] += 1
                    if s.features.feature['label'].int64_list.value[0] == guessed_label:
                        a+=1
                    f['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[guessed_label]))
                    f['index'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[ori_label[pos]]))
                    s = tf.train.Example(features=tf.train.Features(feature=f))
                    s = s.SerializeToString()
                    writer_label.write(s)
                    total += 1
                pos += 1
                loop.update()
        loop.close()
        print('N', N)
        print(1.*a/len(label), a, len(label))
        print('total', total)
    with tf.gfile.Open(target + '-label.json', 'w') as writer:
        writer.write(json.dumps(dict(distribution=train_stats.tolist(), label=sorted(label)), indent=2, sort_keys=True))


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
