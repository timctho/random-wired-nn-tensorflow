import tensorflow as tf
from src.model.base_net import RandWire
from src.model.custom_layer import RandWireLayer


class SmallRandWireNN(RandWire):
    def __init__(self, config, is_training=True):
        super(SmallRandWireNN, self).__init__(config, is_training)

        if config['Graph']['dag_def']:
            graph_def = config['Graph']['dag_def']
        else:
            graph_def = [None, None, None]

        self.rand_wire_layer_0 = RandWireLayer(
            self.base_channel, self.base_channel, self.n, self.k, self.p, self.m,
            graph_def[0], graph_mode=config['Graph']['mode'], name='RandWire_0', is_training=is_training)

        self.rand_wire_layer_1 = RandWireLayer(
            self.base_channel, 2 * self.base_channel, self.n, self.k, self.p, self.m,
            graph_def[1], graph_mode=config['Graph']['mode'], name='RandWire_1', is_training=is_training)

        self.rand_wire_layer_2 = RandWireLayer(
            2 * self.base_channel, 4 * self.base_channel, self.n, self.k, self.p, self.m,
            graph_def[2], graph_mode=config['Graph']['mode'], name='RandWire_2', is_training=is_training)

    def __call__(self, inputs):
        with tf.variable_scope('Conv1'):
            out = tf.layers.conv2d(inputs, self.base_channel // 2, 3, 2, 'same')
            out = tf.layers.batch_normalization(out, training=self.is_training)

        with tf.variable_scope('Conv2'):
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, self.base_channel, 3, 1, 'same')
            out = tf.layers.batch_normalization(out, training=self.is_training)

        out = self.rand_wire_layer_0(out)
        out = self.rand_wire_layer_1(out)
        out = self.rand_wire_layer_2(out)

        with tf.variable_scope('Classifier'):
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, 1280, 1)
            out = tf.layers.batch_normalization(out, training=self.is_training)
            out = tf.reduce_mean(out, axis=[1, 2], keep_dims=True)
            out = tf.layers.dense(out, self.num_class)
        return out
