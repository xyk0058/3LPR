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

import numpy as np
from absl import app
from absl import flags

from ablation.ab_cta_remixmatch import ABCTAReMixMatch
from libml import utils, data, ctaugment
from libml.augment import AugmentPoolCTA

FLAGS = flags.FLAGS


class AugmentPoolCTANoWeak(AugmentPoolCTA):
    @staticmethod
    def numpy_apply_policies(arglist):
        x, cta, probe = arglist
        if x.ndim == 3:
            assert probe
            policy = cta.policy(probe=True)
            return dict(policy=policy,
                        probe=ctaugment.apply(x, policy),
                        image=ctaugment.apply(x, cta.policy(probe=False)))
        assert not probe
        return dict(image=np.stack([ctaugment.apply(y, cta.policy(probe=False)) for y in x]).astype('f'))


class ABCTAReMixMatchNoWeak(ABCTAReMixMatch):
    AUGMENT_POOL_CLASS = AugmentPoolCTANoWeak


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = data.MANY_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = ABCTAReMixMatchNoWeak(
        os.path.join(FLAGS.train_dir, dataset.name, ABCTAReMixMatchNoWeak.cta_name()),
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
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


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
    app.run(main)
