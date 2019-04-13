import tensorflow as tf
import logging
import os
from absl import app, flags
import horovod.tensorflow as hvd

from src.model.model_fn import horovod_model_fn
from src.model.util import *
from src.util.misc import load_config

flags.DEFINE_string('config', 'src/config/small.json',
                    help='Path to config file')
flags.DEFINE_string('save_path', 'small_imagenet',
                    help='Path to save ckpt and logging files')
flags.DEFINE_string('pretrained', '',
                    help='Continue training from this pretrained model')
FLAGS = flags.FLAGS


def main(_):
    hvd.init()

    config = load_config(FLAGS.config)
    config['training_graph_path'] = os.path.join(FLAGS.save_path, 'rand_graph')
    config['save_path'] = FLAGS.save_path

    dataset_name = config['Data']['name']
    if dataset_name == 'mnist':
        num_train_images = mnist.MNIST_SIZE
    elif dataset_name == 'imagenet':
        num_train_images = imagenet.NUM_IMAGES['train']
    config['Train']['cos_lr']['step'] = num_train_images * config['Train']['epoch'] // \
                                        config['Data']['batch_size']

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

    estimator_config = tf.estimator.RunConfig(session_config=sess_config,
                                              log_step_count_steps=config['Monitor']['log_step'],
                                              keep_checkpoint_max=100,
                                              save_checkpoints_steps=config['Monitor'][
                                                                         'ckpt_save_epoch'] * num_train_images //
                                                                     config['Data']['batch_size'])

    hooks = [
        hvd.BroadcastGlobalVariablesHook(0)
    ]

    estimator = tf.estimator.Estimator(
        model_fn=horovod_model_fn,
        model_dir=FLAGS.save_path if hvd.rank() == 0 else None,
        config=estimator_config,
        params=config)

    for _ in range(config['Train']['epoch'] // config['Monitor']['ckpt_save_epoch'] // hvd.size()):
        estimator.train(input_fn=lambda: get_train_dataset(config), hooks=hooks)
        if hvd.rank() == 0:
            estimator.evaluate(input_fn=lambda: get_eval_dataset(config))


if __name__ == '__main__':
    logging.getLogger('tensorflow').propagate = False
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
