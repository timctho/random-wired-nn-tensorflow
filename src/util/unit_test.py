import sys
sys.path.append('../../')

import tensorflow as tf
from src.model.small_regime import SmallRandWireNN
from src.util.misc import load_config


if __name__ == '__main__':
    config = load_config('../../src/config/default.json')
    ph = tf.placeholder(tf.float32, [1, 224, 224, 3])
    model = SmallRandWireNN(config, False)
    out = model(ph)

    board = tf.summary.FileWriter('./', graph=tf.get_default_graph())