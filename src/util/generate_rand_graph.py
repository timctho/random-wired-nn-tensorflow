import sys

sys.path.append('./')

from src.util.misc import load_config
from src.model.small_regime import SmallRandWireNN
from src.model.regular_regime import RegularRandWireNN
from src.model.mnist_regime import MnistRandWireNN

from absl import flags, app
import os

flags.DEFINE_string('config', '', help='Path to config')
flags.DEFINE_string('save_path', '', help='Path to ckpt')
FLAGS = flags.FLAGS


def main(_):
    config = load_config(FLAGS.config)
    config['training_graph_path'] = os.path.join(FLAGS.save_path, 'rand_graph')

    arch = config['Model']['arch']
    if arch == 'small':
        model = SmallRandWireNN(config, True)
    elif arch == 'regular':
        model = RegularRandWireNN(config, True)
    elif arch == 'mnist':
        model = MnistRandWireNN(config, True)
    else:
        raise ValueError('Not recognized model architecture {}'.format(arch))


if __name__ == '__main__':
    app.run(main)
