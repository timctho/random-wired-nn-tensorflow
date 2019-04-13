import tensorflow as tf
import os
from absl import app, flags
import logging

from src.model.model_fn import model_fn
from src.model.util import *
from src.util.misc import load_config
from src.util import distribute

flags.DEFINE_string('config', 'src/config/default.json',
                    help='Path to config file')
flags.DEFINE_integer('num_gpu', 1,
                     help='If greater or equal to 2, use distribute training')
flags.DEFINE_string('save_path', '',
                    help='Path to save ckpt and logging files')
flags.DEFINE_string('pretrained', '',
                    help='Continue training from this pretrained model')
FLAGS = flags.FLAGS


def main(_):
    config = load_config(FLAGS.config)
    config['training_graph_path'] = os.path.join(FLAGS.save_path, 'rand_graph')
    config['save_path'] = FLAGS.save_path

    config['Train']['cos_lr']['step'] = get_total_train_iters(config)

    if FLAGS.num_gpu > 1:
        # Creates session config. allow_soft_placement = True, is required for
        # multi-GPU and is not harmful for other modes.
        session_config = tf.ConfigProto(allow_soft_placement=True)

        distribution_strategy = distribute.get_distribution_strategy(
            distribution_strategy='mirrored',
            num_gpus=FLAGS.num_gpu)

        # Creates a `RunConfig` that checkpoints every 24 hours which essentially
        # results in checkpoints determined only by `epochs_between_evals`.
        run_config = tf.estimator.RunConfig(
            train_distribute=distribution_strategy,
            session_config=session_config)
    else:
        run_config = tf.estimator.RunConfig(log_step_count_steps=config['Monitor']['log_step'])

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.save_path,
        config=run_config,
        params=config)

    for _ in range(config['Train']['epoch'] // config['Monitor']['ckpt_save_epoch']):
        estimator.train(input_fn=lambda: get_train_dataset(config))
        estimator.evaluate(input_fn=lambda: get_eval_dataset(config))


if __name__ == '__main__':
    logging.getLogger('tensorflow').propagate = False
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
