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
"""ReMixMatch training, changes from MixMatch are:
- Add distribution matching.
"""

import os

from absl import app
from absl import flags

from cta.lib.train import CTAClassifySemi
from libml import data, utils
from newmixmatch import NewMixMatch
import numpy as np
import tensorflow as tf
from tqdm import trange

FLAGS = flags.FLAGS


class CTANewMixMatch(NewMixMatch, CTAClassifySemi):
    
    def train_step(self, train_session, gen_labeled, gen_unlabeled):
        x, y = gen_labeled(), gen_unlabeled()
        y_index = y['index'][:,0]
        x_index = x['index']
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

        # print('labeled_idx', x['probe'].shape, x['image'].shape, y['image'].shape)

        # x, y dict_keys(['image', 'label', 'index', 'cta', 'probe', 'policy']) dict_keys(['image', 'label', 'index', 'cta', 'probe'])
        v = train_session.run([self.ops.classify_op, self.ops.train_op, self.ops.ly, self.ops.update_step]+self.ops.l_dbg,
                              feed_dict={self.ops.y: y['image'],
                                         self.ops.x: x['probe'],
                                         self.ops.xt: x['image'],
                                         self.ops.label: x['label'],
                                         self.ops.Number: N,
                                         self.ops.Z_labeled: Z_labeled,
                                         self.ops.Z_unlabeled: Z_unlabeled,
                                         self.ops.sample_epoch: self.sample_epoch[y_index]})

        # v0 = []
        # for i in v[-3]:
        #     # for j in i:
        #     if i <= 0:
        #         v0.append(i)
        # if len(v0) > 0:
        #     print('v0', v0)
        # print('v1', v[-2])
        if v[3] % (FLAGS.report_kimg << 10) == 0:
            print('w_match', v[-1], v[3])
        # Soft
        gfy = v[2][:FLAGS.batch]
        z_old = np.sum(self.z[y_index], axis=0)
        self.Z[y_index] = FLAGS.ema_beta * self.Z[y_index] + (1 - FLAGS.ema_beta) * gfy

        self.Z[self.dataset.labeled_index] = np.eye(self.nclass)[self.dataset.labels]

        self.z[y_index] = self.Z[y_index] * (1. / (1. - FLAGS.ema_beta **
                       (self.sample_epoch[y_index] + 1)))
        self.z[self.dataset.labeled_index] = np.eye(self.nclass)[self.dataset.labels]

        self.sample_epoch[y_index] = self.sample_epoch[y_index] + 1

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

        
            # it = self.dataset.train_labeled.parse().augment().batch(1).prefetch(16).make_one_shot_iterator().get_next()
            # while 1:
            #     try:
            #         v = self.session.run(it)
            #         if v['image'].shape == (1, 32, 32, 3):
            #             print('v!!!!!!!!!!!', v['image'].shape)
            #     except tf.errors.OutOfRangeError:
            #         break
            # breaks


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

            log_file = os.path.join(FLAGS.train_dir, self.dataset.name, CTANewMixMatch.cta_name())
            if not os.path.exists(log_file):
                os.makedirs(log_file)
            with open(log_file+'/log.out', 'w', buffering=1) as log_stream:
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
    # dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
    nclass = 10
    print('FLAGS.dataset.split('')[0]', FLAGS.dataset.split('.')[0])
    if FLAGS.dataset.split('.')[0] == 'cifar100':
        nclass = 100
    # dataset = data.DataSets.creator(FLAGS.dataset, nclass=nclass)[1]()
    dataset = data.DataSets.creator(FLAGS.dataset, nclass=nclass, height=32, width=32)[1]()
    log_width = utils.ilog2(dataset.width)
    model = CTANewMixMatch(
        os.path.join(FLAGS.train_dir, dataset.name, CTANewMixMatch.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        beta=FLAGS.beta,
        w_match=FLAGS.w_match,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.75, 'Mixup beta distribution.')
    flags.DEFINE_float('w_match', 100, 'Weight for distribution matching loss.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 16)

    flags.DEFINE_float('ema_beta', 0.6, 'ema_beta.')
    FLAGS.set_default('ema_beta', 0.6)
    flags.DEFINE_float('imb_fac', 10, 'imbalanced ratio.')
    FLAGS.set_default('imb_fac', 10)
    flags.DEFINE_string('log_root', './log/', help='log file root')
    FLAGS.set_default('log_root', './log/')

    app.run(main)
