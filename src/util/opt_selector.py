import tensorflow as tf


class OptimizerSelector(object):
    def __init__(self, config, lr):
        self._optimizer = None

        if config['Train']['optimizer'].lower() == 'sgd':
            self._optimizer = tf.train.MomentumOptimizer(lr, 0.9)

        elif config['Train']['optimizer'].lower() == 'adam':
            self._optimizer = tf.train.AdamOptimizer(lr)

        elif config['Train']['optimizer'].lower() == 'rmsprop':
            self._optimizer = tf.train.RMSPropOptimizer(lr)

    @property
    def optimizer(self):
        return self._optimizer
