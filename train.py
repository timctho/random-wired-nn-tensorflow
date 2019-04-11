import tensorflow as tf
import json

from src.model.small_regime import SmallRandWireNN
from src.model.regular_regime import RegularRandWireNN
from src.util.lr_selector import LearningRateSelector
from src.util.opt_selector import OptimizerSelector

from absl import app, flags

flags.DEFINE_string('config', 'src/config/default.json',
                    help='Path to config file')

flags.DEFINE_string('gpu', '0',
                    help='GPU ID used')

flags.DEFINE_string('save_path', '',
                    help='Path to save ckpt and logging files')

flags.DEFINE_string('pretrained', '',
                    help='Continue training from this pretrained model')

FLAGS = flags.FLAGS
