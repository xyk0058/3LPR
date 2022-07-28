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
"""Custom neural network layers and primitives.
"""
import numbers

import numpy as np
import tensorflow as tf

from libml.data import DataSets


def smart_shape(x):
    s, t = x.shape, tf.shape(x)
    return [t[i] if s[i].value is None else s[i] for i in range(len(s))]


def entropy_from_logits(logits):
    """Computes entropy from classifier logits.

    Args:
        logits: a tensor of shape (batch_size, class_count) representing the
        logits of a classifier.

    Returns:
        A tensor of shape (batch_size,) of floats giving the entropies
        batchwise.
    """
    distribution = tf.contrib.distributions.Categorical(logits=logits)
    return distribution.entropy()


def entropy_penalty(logits, entropy_penalty_multiplier, mask):
    """Computes an entropy penalty using the classifier logits.

    Args:
        logits: a tensor of shape (batch_size, class_count) representing the
            logits of a classifier.
        entropy_penalty_multiplier: A float by which the entropy is multiplied.
        mask: A tensor that optionally masks out some of the costs.

    Returns:
        The mean entropy penalty
    """
    entropy = entropy_from_logits(logits)
    losses = entropy * entropy_penalty_multiplier
    losses *= tf.cast(mask, tf.float32)
    return tf.reduce_mean(losses)


def kl_divergence_from_logits(logits_a, logits_b):
    """Gets KL divergence from logits parameterizing categorical distributions.

    Args:
        logits_a: A tensor of logits parameterizing the first distribution.
        logits_b: A tensor of logits parameterizing the second distribution.

    Returns:
        The (batch_size,) shaped tensor of KL divergences.
    """
    distribution1 = tf.contrib.distributions.Categorical(logits=logits_a)
    distribution2 = tf.contrib.distributions.Categorical(logits=logits_b)
    return tf.contrib.distributions.kl_divergence(distribution1, distribution2)


def mse_from_logits(output_logits, target_logits):
    """Computes MSE between predictions associated with logits.

    Args:
        output_logits: A tensor of logits from the primary model.
        target_logits: A tensor of logits from the secondary model.

    Returns:
        The mean MSE
    """
    diffs = tf.nn.softmax(output_logits) - tf.nn.softmax(target_logits)
    squared_diffs = tf.square(diffs)
    return tf.reduce_mean(squared_diffs, -1)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [tf.concat(v, axis=0) for v in xy]


def renorm(v):
    return v / tf.reduce_sum(v, axis=-1, keepdims=True)


def shakeshake(a, b, training):
    if not training:
        return 0.5 * (a + b)
    mu = tf.random_uniform([tf.shape(a)[0]] + [1] * (len(a.shape) - 1), 0, 1)
    mixf = a + mu * (b - a)
    mixb = a + mu[::1] * (b - a)
    return tf.stop_gradient(mixf - mixb) + mixb


class PMovingAverage:
    def __init__(self, name, nclass, buf_size):
        # MEAN aggregation is used by DistributionStrategy to aggregate
        # variable updates across shards
        self.ma = tf.Variable(tf.ones([buf_size, nclass]) / nclass,
                              trainable=False,
                              name=name,
                              aggregation=tf.VariableAggregation.MEAN)

    def __call__(self):
        v = tf.reduce_mean(self.ma, axis=0)
        return v / tf.reduce_sum(v)

    def update(self, entry):
        entry = tf.reduce_mean(entry, axis=0)
        return tf.assign(self.ma, tf.concat([self.ma[1:], [entry]], axis=0))



class PData:
    def __init__(self, dataset: DataSets):
        self.has_update = False
        if dataset.p_unlabeled is not None:
            self.p_data = tf.constant(dataset.p_unlabeled, name='p_data')
        elif dataset.p_labeled is not None:
            self.p_data = tf.constant(dataset.p_labeled, name='p_data')
        else:
            # MEAN aggregation is used by DistributionStrategy to aggregate
            # variable updates across shards
            self.p_data = tf.Variable(renorm(tf.ones([dataset.nclass])),
                                      trainable=False,
                                      name='p_data',
                                      aggregation=tf.VariableAggregation.MEAN)
            self.has_update = True

    def __call__(self):
        return self.p_data / tf.reduce_sum(self.p_data)

    def update(self, entry, decay=0.999):
        entry = tf.reduce_mean(entry, axis=0)
        return tf.assign(self.p_data, self.p_data * decay + entry * (1 - decay))


class MixMode:
    # A class for mixing data for various combination of labeled and unlabeled.
    # x = labeled example
    # y = unlabeled example
    # For example "xx.yxy" means: mix x with x, mix y with both x and y.
    MODES = 'xx.yy xxy.yxy xx.yxy xx.yx xx. .yy xxy. .yxy .'.split()

    def __init__(self, mode):
        assert mode in self.MODES
        self.mode = mode

    # @staticmethod
    # def augment_pair(x0, l0, x1, l1, beta, N, **kwargs):
    #     N, imb_fac, Z_unlabeled, Z_labeled, ema_beta, sample_epoch = N[0], N[1], N[2], N[3], N[4], N[5]
    #     del kwargs
    #     if isinstance(beta, numbers.Integral) and beta <= 0:
    #         return x0, l0

    #     def np_beta(s, beta):  # TF implementation seems unreliable for beta below 0.2
    #         return np.random.beta(beta, beta, s).astype('f')

    #     with tf.device('/cpu'):
    #         mix = tf.py_func(np_beta, [tf.shape(x0)[0], beta], tf.float32)
    #         mix = tf.reshape(tf.maximum(mix, 1 - mix), [tf.shape(x0)[0], 1, 1, 1])
    #         index = tf.random_shuffle(tf.range(tf.shape(x0)[0]))
    #     xs = tf.gather(x1, index)
    #     ls = tf.gather(l1, index)

    #     N0 = tf.gather(N, tf.argmax(l0, 1))
    #     NS = tf.gather(N, tf.argmax(ls, 1))

    #     # xmix = x0 * mix + xs * (1 - mix)
    #     # lmix = l0 * mix[:, :, 0, 0] + ls * (1 - mix[:, :, 0, 0])
    #     # x00 = x0 * mix + xs * (1. - mix)
    #     x01 = xs * mix + x0 * (1. - mix)
    #     # xmix = tf.where(tf.less(N0, NS), x01, x01)
    #     # l00 = l0 * mix[:, :, 0, 0] + ls * (1. - mix[:, :, 0, 0])
    #     l01 = ls * mix[:, :, 0, 0] + l0 * (1. - mix[:, :, 0, 0])
    #     # lmix = tf.where(tf.less(N0, NS), l01, l01)
    #     xmix = x01
    #     lmix = l01

    #     return xmix, lmix, [xmix, tf.where(N0>NS), N0, NS, l0]#(640, 32, 32, 3) (200, 1) (640,) (640,) (640, 10)
    
    @staticmethod
    def augment_pair(x0, l0, x1, l1, beta, N, **kwargs):
        N, imb_fac, Z_unlabeled, Z_labeled, ema_beta, sample_epoch = N[0], N[1], N[2], N[3], N[4], N[5]
        del kwargs
        if isinstance(beta, numbers.Integral) and beta <= 0:
            return x0, l0

        def np_beta(s, alpha, beta):  # TF implementation seems unreliable for beta below 0.2
            return np.random.beta(alpha, beta, s).astype('f')
        
        def ramp_up(epoch, max_epochs=100, max_val=1., mult=-5.):
            return max_val * tf.exp(mult * (1. - epoch / max_epochs) ** 2)

        Z = tf.concat([Z_labeled, Z_unlabeled], axis=0)

        index = tf.random_shuffle(tf.range(tf.shape(x0)[0]))
        xs = tf.gather(x1, index)
        ls = tf.gather(l1, index)
        Zs = tf.gather(Z, index)

        sample_epoch = sample_epoch[0]

        l0_ = ema_beta * Z + (1. - ema_beta) * l0
        l0_new = l0_ * (1. / (1. - ema_beta ** (sample_epoch + 1)))
        ls_ = ema_beta * Zs + (1. - ema_beta) * ls
        ls_new = ls_ * (1. / (1. - ema_beta ** (sample_epoch + 1)))
        Na = tf.reverse(-tf.log( tf.abs((N / tf.reduce_sum(N)) ** 1 + 1e-12) ), [0])

        N0 = tf.pow(tf.abs(tf.gather(Na, tf.argmax(l0_new, 1))), 1)
        NS = tf.pow(tf.abs(tf.gather(Na, tf.argmax(ls_new, 1))), 1)

        tmp = tf.where(N0<NS, NS / N0, N0 / NS) - 1

        # confidence = (-tf.reduce_sum(l0 * tf.log(tf.abs(l0+1e-12)), axis=1)+1) / (-tf.reduce_sum(ls * tf.log(tf.abs(ls+1e-12)), axis=1)+1)
        # confidence_tmp = tf.where(N0<NS, -tf.ones(tf.shape(confidence)), tf.ones(tf.shape(confidence)))

        # confidence = tf.where(N0<NS, -tf.reduce_sum(l0_new * tf.log(tf.abs(l0_new+1e-12)), axis=1)+1, (-tf.reduce_sum(ls_new * tf.log(tf.abs(ls_new+1e-12)), axis=1)+1))

        # confidence_ = tf.pow(confidence, confidence_tmp)
        # # confidence = tf.tanh(confidence)
        # confidencee = tf.pow(confidence_, 1./4)
        # rho = 1. + confidencee * (tmp/(Na[0]/Na[9]))
        # #rho = 1. + confidence * (tmp / (imb_fac+10000000000))

        #confidence = 1. + tf.where(N0<NS, -tf.reduce_sum(l0 * tf.log(tf.abs(l0+1e-12)), axis=1), -tf.reduce_sum(ls * tf.log(tf.abs(ls+1e-12)), axis=1))
        # confidence = 1 + tf.where(N0<NS, tf.reduce_max(l0_new, axis=1), tf.reduce_max(ls_new, axis=1))

        # l0_new_ = tf.where((l0_new+1e-12)>1, tf.ones(tf.shape(l0_new)), l0_new)
        # ls_new_ = tf.where((l0_new+1e-12)>1, tf.ones(tf.shape(ls_new)), ls_new)
        l0_new_ = l0_new
        ls_new_ = ls_new

        confidence = 1. + tf.where(N0<NS, -tf.reduce_sum(l0_new_ * tf.log(tf.abs(l0_new_+1e-12)), axis=1), -tf.reduce_sum(ls_new_ * tf.log(tf.abs(ls_new_+1e-12)), axis=1))
        rho_ = 1. + confidence * (tmp/(Na[0]/Na[-1]))
        rho = rho_
        #rho = tf.where(rho_>2, 2*tf.ones(tf.shape(rho_)), rho_)

        # with tf.device('/cpu'):
        #      lam_ = tf.py_func(np_beta, [tf.shape(x0)[0], beta, beta], tf.float32)
        #      lam_ = tf.reshape(tf.maximum(lam_, 1 - lam_), [tf.shape(x0)[0], 1, 1, 1])

        with tf.device('/cpu'):
            lam_l_ = tf.py_func(np_beta, [tf.shape(Z_labeled)[0], beta, beta], tf.float32)
            lam_l_ = tf.reshape(tf.maximum(lam_l_, 1 - lam_l_), [tf.shape(Z_labeled)[0], 1, 1, 1])
            lam_u_ = tf.py_func(np_beta, [tf.shape(Z_unlabeled)[0], tf.reshape(rho[tf.shape(Z_labeled)[0]:], [tf.shape(Z_unlabeled)[0]]), tf.ones(tf.shape(Z_unlabeled)[0])], tf.float32)
            # lam_u_ = tf.py_func(np_beta, [tf.shape(Z_unlabeled)[0], beta, beta], tf.float32)

            lam_u_ = tf.reshape(tf.where(N0[tf.shape(Z_labeled)[0]:]<NS[tf.shape(Z_labeled)[0]:], lam_u_, 1 - lam_u_), [tf.shape(Z_unlabeled)[0], 1, 1, 1])
            lam_ = tf.concat([lam_l_, lam_u_], axis=0)

        xmix = x0 * lam_ + xs * (1. - lam_)
        lam_lra_l = lam_[:tf.shape(Z_labeled)[0], :, 0, 0]
        lmix_l = l0_new[:tf.shape(Z_labeled)[0]] * lam_lra_l + ls_new[:tf.shape(Z_labeled)[0]] * (1 - lam_lra_l)
        lam = tf.where(N0<NS, lam_, 1.-lam_)

        lam_lra = tf.reshape(rho, [tf.shape(x0)[0], 1, 1, 1]) * lam
        lam_lra = tf.where(lam_lra>1, tf.ones(lam_lra.shape), lam_lra)

        lam_lra = tf.where(N0<NS, lam_lra, 1.-lam_lra)
        lmix_u = l0_new[tf.shape(Z_labeled)[0]:]  * lam_lra[tf.shape(Z_labeled)[0]:, :, 0, 0] + ls_new[tf.shape(Z_labeled)[0]:] * (1 - lam_lra[tf.shape(Z_labeled)[0]:, :, 0, 0])
        lmix = tf.concat([lmix_l, lmix_u], axis=0)



        #lmix = l0 * lam_lra[:, :, 0, 0] + ls * (1 - lam_lra[:, :, 0, 0])
        #lmix = l0 * lam_[:, :, 0, 0] + ls * (1 - lam_[:, :, 0, 0])

        # confidence
        # lam_lra_l = lam_[:tf.shape(Z_labeled)[0], :, 0, 0]
        # lmix_l = l0[:tf.shape(Z_labeled)[0]] * lam_lra_l + ls[:tf.shape(Z_labeled)[0]] * (1 - lam_lra_l)

        # confidence = (-tf.reduce_sum(ls[tf.shape(Z_labeled)[0]:] * tf.log(tf.abs(ls[tf.shape(Z_labeled)[0]:]+1e-12)), axis=1)+1) / (-tf.reduce_sum(l0[tf.shape(Z_labeled)[0]:] * tf.log(tf.abs(l0[tf.shape(Z_labeled)[0]:]+1e-12)), axis=1)+1)
        # w = 2
        # confidence = tf.tanh(w * confidence)
        # confidence = tf.reshape(confidence, [tf.shape(confidence)[0], 1])
        # lmix_u = l0[tf.shape(Z_labeled)[0]:] * confidence * lam_lra[tf.shape(Z_labeled)[0]:, :, 0, 0] + ls[tf.shape(Z_labeled)[0]:] * (1 - confidence * lam_lra[tf.shape(Z_labeled)[0]:, :, 0, 0])
        # lmix = tf.concat([lmix_l, lmix_u], axis=0)

        # confidence ranking
        # lam_lra_l = lam_[:tf.shape(Z_labeled)[0], :, 0, 0]
        # lmix_l = l0[:tf.shape(Z_labeled)[0]] * lam_lra_l + ls[:tf.shape(Z_labeled)[0]] * (1 - lam_lra_l)

        # entropy_l0 = -tf.reduce_sum(l0[tf.shape(Z_labeled)[0]:] * tf.log(tf.abs(l0[tf.shape(Z_labeled)[0]:]+1e-12)), axis=1) 
        # l0_sorted_idx = tf.argsort(entropy_l0, direction='DESCENDING')
        # rank = tf.cast(tf.range(tf.shape(Z_unlabeled)[0]), tf.float32)
        # # b = tf.cast(tf.shape(Z_unlabeled)[0] / 2, tf.float32)
        # b = 64. * 9. / 2.
        # w = 1000
        # c = tf.sigmoid(w * ((rank - b) / tf.cast(tf.shape(Z_unlabeled)[0], tf.float32)))
        
        # l0_u_sort = tf.gather(l0[tf.shape(Z_labeled)[0]:], l0_sorted_idx)
        # ls_u_sort = tf.gather(ls[tf.shape(Z_labeled)[0]:], l0_sorted_idx)
        # c_sort = tf.gather(c, l0_sorted_idx)
        # c_sort = tf.reshape(c_sort, [tf.shape(c_sort)[0], 1])
        # lam_lra_u = tf.gather(lam_lra[tf.shape(Z_labeled)[0]:, :, 0, 0], l0_sorted_idx)
        # lmix_u = l0_u_sort * c_sort * lam_lra_u + ls_u_sort * (1 - c_sort * lam_lra_u)

        # lmix = tf.concat([lmix_l, lmix_u], axis=0)

        # reweight
        #lam_lra_l = lam_[:tf.shape(Z_labeled)[0], :, 0, 0]
        #lmix_l = l0[:tf.shape(Z_labeled)[0]] * lam_lra_l + ls[:tf.shape(Z_labeled)[0]] * (1 - lam_lra_l)

        #Na = -tf.log( tf.abs((N / tf.reduce_sum(N)) ** 0.5 + 1e-12) )
        #N0 = tf.gather(Na, tf.argmax(l0, 1))
        #NS = tf.gather(Na, tf.argmax(ls, 1))
        #W0 = N0[tf.shape(Z_labeled)[0]:]
        #W0 = tf.reshape(W0, [tf.shape(W0)[0], 1])

        #WS = NS[tf.shape(Z_labeled)[0]:]
        #WS = tf.reshape(WS, [tf.shape(WS)[0], 1])

        #lmix_u = l0[tf.shape(Z_labeled)[0]:] * W0 * lam_lra[tf.shape(Z_labeled)[0]:, :, 0, 0] + ls[tf.shape(Z_labeled)[0]:] * (1 - lam_lra[tf.shape(Z_labeled)[0]:, :, 0, 0]) * WS

        #lmix = tf.concat([lmix_l, lmix_u], axis=0)

        #reweighting2
        #Na = -tf.log( tf.abs((N / tf.reduce_sum(N)) ** 0.5 + 1e-12) )
        #W0 = tf.reshape(tf.gather(Na, tf.argmax(l0, 1)), [tf.shape(l0)[0], 1])
        #WS = tf.reshape(tf.gather(Na, tf.argmax(ls, 1)), [tf.shape(ls)[0], 1])

        #lmix = l0 * W0 * lam_[:, :, 0, 0] + ls * (1 - lam_[:, :, 0, 0]) * WS 



        return xmix, lmix, [Z, Z_labeled, lam, lam_lra, rho[tf.shape(Z_labeled)[0]:], ls_new, confidence]

    @staticmethod
    def augment(x, l, beta, N=None, **kwargs):
        return MixMode.augment_pair(x, l, x, l, beta, N, **kwargs)

    def __call__(self, xl: list, ll: list, betal: list, N=None):
        assert len(xl) == len(ll) >= 2
        assert len(betal) == 2
        if self.mode == '.':
            return xl, ll
        elif self.mode == 'xx.':
            mx0, ml0 = self.augment(xl[0], ll[0], betal[0], N)
            return [mx0] + xl[1:], [ml0] + ll[1:]
        elif self.mode == '.yy':
            mx1, ml1 = self.augment(
                tf.concat(xl[1:], 0), tf.concat(ll[1:], 0), betal[1], N)
            return (xl[:1] + tf.split(mx1, len(xl) - 1),
                    ll[:1] + tf.split(ml1, len(ll) - 1))
        elif self.mode == 'xx.yy':
            mx0, ml0 = self.augment(xl[0], ll[0], betal[0], N)
            mx1, ml1 = self.augment(
                tf.concat(xl[1:], 0), tf.concat(ll[1:], 0), betal[1], N)
            return ([mx0] + tf.split(mx1, len(xl) - 1),
                    [ml0] + tf.split(ml1, len(ll) - 1))
        elif self.mode == 'xxy.':
            mx, ml = self.augment(
                tf.concat(xl, 0), tf.concat(ll, 0),
                sum(betal) / len(betal), N)
            return (tf.split(mx, len(xl))[:1] + xl[1:],
                    tf.split(ml, len(ll))[:1] + ll[1:])
        elif self.mode == '.yxy':
            mx, ml = self.augment(
                tf.concat(xl, 0), tf.concat(ll, 0),
                sum(betal) / len(betal), N)
            return (xl[:1] + tf.split(mx, len(xl))[1:],
                    ll[:1] + tf.split(ml, len(ll))[1:])
        elif self.mode == 'xxy.yxy':
            mx, ml, l_dbg = self.augment(
                tf.concat(xl, 0), tf.concat(ll, 0),
                sum(betal) / len(betal), N)
            return tf.split(mx, len(xl)), tf.split(ml, len(ll)), l_dbg
        elif self.mode == 'xx.yxy':
            mx0, ml0 = self.augment(xl[0], ll[0], betal[0], N)
            mx1, ml1 = self.augment(tf.concat(xl, 0), tf.concat(ll, 0), betal[1], N)
            mx1, ml1 = [tf.split(m, len(xl))[1:] for m in (mx1, ml1)]
            return [mx0] + mx1, [ml0] + ml1
        elif self.mode == 'xx.yx':
            mx0, ml0 = self.augment(xl[0], ll[0], betal[0], N)
            mx1, ml1 = zip(*[
                self.augment_pair(xl[i], ll[i], xl[0], ll[0], betal[1], N)
                for i in range(1, len(xl))
            ])
            return [mx0] + list(mx1), [ml0] + list(ml1)
        raise NotImplementedError(self.mode)
