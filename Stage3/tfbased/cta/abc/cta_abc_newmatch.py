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

import os

from absl import app
from absl import flags

from cta.lib.train import CTAClassifySemi
from libml import utils, data
from cta.abc.newmatch_abc import ABCNewMatch

import json
import os.path
import shutil

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm
from libml.utils import EasyDict

from collections import Counter


FLAGS = flags.FLAGS
flags.DEFINE_string('ir2', '', 'ir2 in ABC.')

class ABCCTANewMatch(ABCNewMatch, CTAClassifySemi):

    def train_step(self, train_session, gen_labeled, gen_unlabeled):
        x, y = gen_labeled(), gen_unlabeled()
        y_index = y['index'][:,0]
        x_index = x['index']

        if FLAGS.use_abc:
            sequence = FLAGS.ir2.split(',')
            ir2 = np.array(list(map(float, sequence)), dtype=np.float32)
            ir2 = ir2[-1] / ir2
            # print('ir2', ir2, FLAGS.ir2)
        else:
            ir2 = [1 for _ in range(self.nclass)]


        N = self.N_label + self.N_unlabel

        Z_unlabeled = self.Z[y_index]
        Z_labeled = self.Z[x_index]
        
        labeled_idx = []
        labeled_label = []
        for idx in y_index:
            labeled_idx.append(idx)
            one_hot = self.Z[idx]
            labeled_label.append(one_hot)
        labeled_idx = np.array(labeled_idx)
        labeled_label = np.array(labeled_label)
        # print('labeled_idx', labeled_idx.shape, labeled_label.shape)

        # x, y dict_keys(['image', 'label', 'index', 'cta', 'probe', 'policy']) dict_keys(['image', 'label', 'index', 'cta', 'probe'])
        v = train_session.run([self.ops.classify_op, self.ops.train_op, self.ops.ly, self.ops.update_step]+self.ops.l_dbg,
                              feed_dict={self.ops.y: y['image'],
                                         self.ops.x: x['probe'],
                                         self.ops.xt: x['image'],
                                         self.ops.label: x['label'],
                                         self.ops.N: N,
                                         self.ops.Z_labeled: Z_labeled,
                                         self.ops.Z_unlabeled: Z_unlabeled,
                                         self.ops.sample_epoch: self.sample_epoch[y_index],
                                         self.ops.labeled_idx: labeled_idx,
                                         self.ops.labeled_label: labeled_label,
                                         self.ops.ir2: ir2})
        # update N in paper Equation (11)
        # print('train_step rho', v[2].shape, v[4].shape, v[5].shape)
        # xs, ls, x0, l0, x1, l1
        # (640, 32, 32, 3) (640, 10) (640, 32, 32, 3) (640, 10) (640, 32, 32, 3) (640, 10)

        # HARD
        # gfy = v[2][:FLAGS.batch]
        # Z_old = np.argmax(self.z[y_index], axis=1)
        # N_old = Counter(Z_old)
        # self.Z[y_index] = FLAGS.ema_beta * self.Z[y_index] + (1 - FLAGS.ema_beta) * gfy
        # self.z[y_index] = self.Z[y_index] * (1. / (1. - FLAGS.ema_beta **
        #                (self.sample_epoch[y_index] + 1)))
        # Z_new = np.argmax(self.z[y_index], axis=1)
        # N_new = Counter(Z_new)
        # for i in range(self.nclass):
        #     self.N_unlabel[i] -= N_old[i]
        #     self.N_unlabel[i] += N_new[i]
        # self.sample_epoch[y_index] = self.sample_epoch[y_index] + 1
        # del gfy, Z_old, N_old, Z_new, N_new, Z_unlabeled, Z_labeled

        # ir22, epoch, abcloss, abcloss1, abcloss2
        # print('abcloss2', v[-1])
        # print('abcloss1', v[-2])
        # print('abcloss', v[-3])
        # print('epoch', v[-4])
        # print('ir22', v[-5])

        # Soft
        gfy = v[2][:FLAGS.batch]
        # print('z_old', self.Z[y_index])
        z_old = np.sum(self.z[y_index], axis=0)
        self.Z[y_index] = FLAGS.ema_beta * self.Z[y_index] + (1 - FLAGS.ema_beta) * gfy

        self.Z[self.dataset.labeled_index] = np.eye(self.nclass)[self.dataset.labels]

        self.z[y_index] = self.Z[y_index] * (1. / (1. - FLAGS.ema_beta **
                       (self.sample_epoch[y_index] + 1)))
        self.z[self.dataset.labeled_index] = np.eye(self.nclass)[self.dataset.labels]

        tmp_sum = np.sum(self.z[y_index], axis=1)
        # print(tmp_sum)
        self.sample_epoch[y_index] = self.sample_epoch[y_index] + 1
        # self.z[y_index] = gfy
        z_new = np.sum(self.z[y_index], axis=0)
        self.N_unlabel = self.N_unlabel - z_old + z_new
        del gfy, z_old, z_new, Z_unlabeled, Z_labeled

        self.tmp.step = v[3]
        lx = v[0]
        for p in range(lx.shape[0]):
            error = lx[p]
            error[x['label'][p]] -= 1
            error = np.abs(error).sum()
            self.augmenter.update_rates(x['policy'][p], 1 - 0.5 * error)

    def train(self, train_nimg, report_nimg):
        self.train_nimg = train_nimg
        self.report_nimg = report_nimg
        if FLAGS.eval_ckpt:
            self.eval_checkpoint(FLAGS.eval_ckpt)
            return
        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_labeled = train_labeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
        train_unlabeled = self.dataset.train_unlabeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_unlabeled = train_unlabeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt, pad_step_number=10))

        with tf.Session(config=utils.get_config()) as sess:
            self.session = sess
            self.cache_eval()

        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=self.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.save_kimg << 10,
                save_summaries_steps=report_nimg - batch) as train_session:
            self.session = train_session._tf_sess()
            gen_labeled = self.gen_labeled_fn(train_labeled)
            gen_unlabeled = self.gen_unlabeled_fn(train_unlabeled)
            self.tmp.step = self.session.run(self.step)

            # N indicate the amount of samples in class
            self.N_label = self.dataset.N
            self.N_unlabel = np.zeros(self.nclass)
            print('N indicate the amount of samples in class', self.N_label)
            # EMA(g(f(x)))
            self.Z = np.zeros((self.dataset.unlabeled_num+self.dataset.labeled_num, self.nclass))
            self.z = np.zeros((self.dataset.unlabeled_num+self.dataset.labeled_num, self.nclass))
            self.sample_epoch = np.zeros((self.dataset.unlabeled_num+self.dataset.labeled_num, 1))
            self.Z[self.dataset.labeled_index] = np.eye(self.nclass)[self.dataset.labels]
            self.z[self.dataset.labeled_index] = np.eye(self.nclass)[self.dataset.labels]

            print('EMA(g(f(x)))', self.Z.shape, np.eye(self.nclass)[self.dataset.labels].shape)
            # EMA(g(f(x))) (14004, 10) (1598, 10)
            # log_file = os.path.join(FLAGS.log_root, self.dataset.name, CTANewMatch.cta_name())
            log_file = os.path.join(FLAGS.train_dir, self.dataset.name, ABCCTANewMatch.cta_name())
            if not os.path.exists(log_file):
                os.makedirs(log_file)
            with open(log_file+'/log.out', 'w') as log_stream:
                while self.tmp.step < train_nimg:
                    loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
                                leave=False, unit='img', unit_scale=batch,
                                desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
                    for _ in loop:
                        self.train_step(train_session, gen_labeled, gen_unlabeled)
                        while self.tmp.print_queue:
                            # loop.write(self.tmp.print_queue.pop(0))
                            s = self.tmp.print_queue.pop(0)
                            loop.write(s)
                            log_stream.write(s+'\r\n')
                while self.tmp.print_queue:
                    # print(self.tmp.print_queue.pop(0))
                    s = self.tmp.print_queue.pop(0)
                    loop.write(s)
                    log_stream.write(s+'\r\n')


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    #dataset = data.MANY_DATASETS()[FLAGS.dataset]()
    nclass = 10
    print('FLAGS.dataset.split('')[0]', FLAGS.dataset.split('.')[0])
    if FLAGS.dataset.split('.')[0] == 'cifar100':
        nclass = 100
    dataset = data.DataSets.creator(FLAGS.dataset, nclass=nclass)[1]()
    log_width = utils.ilog2(dataset.width)
    model = ABCCTANewMatch(
        os.path.join(FLAGS.train_dir, dataset.name, ABCCTANewMatch.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,

        K=FLAGS.K,
        beta=FLAGS.beta,
        w_kl=FLAGS.w_kl,
        w_match=FLAGS.w_match,
        w_rot=FLAGS.w_rot,
        redux=FLAGS.redux,
        use_dm=FLAGS.use_dm,
        use_xe=FLAGS.use_xe,
        warmup_kimg=FLAGS.warmup_kimg,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 9, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('beta', 0.75, 'Mixup beta distribution.')
    flags.DEFINE_float('w_kl', 0.5, 'Weight for KL loss.')
    flags.DEFINE_float('w_match', 1.5, 'Weight for distribution matching loss.')
    flags.DEFINE_float('w_rot', 0.5, 'Weight for rotation loss.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('warmup_kimg', 1024, 'Unannealing duration for SSL loss.')
    flags.DEFINE_enum('redux', '1st', 'swap mean 1st'.split(), 'Logit selection.')
    flags.DEFINE_bool('use_dm', True, 'Whether to use distribution matching.')
    flags.DEFINE_bool('use_xe', True, 'Whether to use cross-entropy or Brier.')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 16)
    flags.DEFINE_bool('use_abc', False, 'Whether to use ABC.')
    flags.DEFINE_float('ema_beta', 0.6, 'ema_beta.')
    FLAGS.set_default('ema_beta', 0.6)
    flags.DEFINE_float('imb_fac', 10, 'imbalanced ratio.')
    FLAGS.set_default('imb_fac', 10)
    flags.DEFINE_string('log_root', './log/', help='log file root')
    FLAGS.set_default('log_root', './log/')
    app.run(main)
