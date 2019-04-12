import tensorflow as tf
import logging
import os

from src.model.small_regime import SmallRandWireNN
from src.model.regular_regime import RegularRandWireNN
from src.model.mnist_regime import MnistRandWireNN

from src.util.lr_selector import LearningRateSelector
from src.util.opt_selector import OptimizerSelector
from src.util.misc import load_config
from src.dataset.mnist import train_input_fn, eval_input_fn

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


def model_fn(features, labels, mode, params):

    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
    else:
        is_training = False

    arch = params['Model']['arch']
    if arch == 'small':
        model = SmallRandWireNN(params, is_training)
    elif arch == 'regular':
        model = RegularRandWireNN(params, is_training)
    elif arch == 'mnist':
        model = MnistRandWireNN(params, is_training)
    else:
        raise ValueError('Not recognized model architecture {}'.format(arch))

    global_step = tf.train.get_or_create_global_step()
    lr = LearningRateSelector(params, global_step).lr
    opt = OptimizerSelector(params, lr).optimizer

    predictions = model(features)

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            loss = tf.losses.sparse_softmax_cross_entropy(labels, predictions)
            train_op = opt.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels, predictions)
        metrics = {
            'Top@1': tf.metrics.accuracy(labels, tf.argmax(predictions, axis=-1)),
            'Top@5': tf.metrics.mean(tf.nn.in_top_k(predictions, labels, 5))
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        _, top_5 = tf.nn.top_k(predictions, k=5)
        results = {
            'Top@1': tf.argmax(predictions, -1),
            'Top@5': top_5,
            'probabilities': tf.nn.softmax(predictions),
            'logits': predictions,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=results)


def main(_):
    config = load_config(FLAGS.config)
    config['training_graph_path'] = os.path.join(FLAGS.save_path, 'rand_graph')

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.save_path,
        params=config)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn('data', config['Data']['batch_size'],
                                                                        config['Monitor']['ckpt_save_epoch']))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn('data', config['Data']['batch_size']))

    for _ in range(config['Train']['epoch'] // config['Monitor']['ckpt_save_epoch']):
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(main)
