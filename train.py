import tensorflow as tf
import logging
import os
from absl import app, flags

from src.dataset import mnist, imagenet
from src.model.small_regime import SmallRandWireNN
from src.model.regular_regime import RegularRandWireNN
from src.model.mnist_regime import MnistRandWireNN

from src.util.lr_selector import LearningRateSelector
from src.util.opt_selector import OptimizerSelector
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


def get_total_loss(predictions, labels, params):
    label_smooth = params['Train']['label_smooth']

    if label_smooth != 0.0:
        one_hot_labels = tf.one_hot(labels, params['Data']['num_class'])
        cls_loss = tf.losses.softmax_cross_entropy(
            logits=predictions, onehot_labels=one_hot_labels, label_smoothing=label_smooth)
    else:
        cls_loss = tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=labels)

    l2_reg_loss = tf.add_n(tf.losses.get_regularization_losses())
    loss = cls_loss + params['Model']['weight_decay'] * l2_reg_loss

    tf.summary.scalar('cls_loss', cls_loss)
    tf.summary.scalar('reg_loss', l2_reg_loss)
    return loss


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

    predictions = model(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        _, top_5 = tf.nn.top_k(predictions, k=5)
        results = {
            'Top@1': tf.argmax(predictions, -1),
            'Top@5': top_5,
            'probabilities': tf.nn.softmax(predictions),
            'logits': predictions,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=results)

    global_step = tf.train.get_or_create_global_step()
    lr = LearningRateSelector(params, global_step).lr
    opt = OptimizerSelector(params, lr).optimizer

    cls_loss = tf.losses.sparse_softmax_cross_entropy(labels, predictions)
    l2_reg_loss = tf.add_n(tf.losses.get_regularization_losses())
    loss = cls_loss + params['Model']['weight_decay'] * l2_reg_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'Top@1': tf.metrics.accuracy(labels, tf.argmax(predictions, axis=-1)),
            'Top@5': tf.metrics.mean(tf.nn.in_top_k(predictions, labels, 5))
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def get_train_dataset(config):
    dataset_name = config['Data']['name']
    if dataset_name == 'mnist':
        return mnist.train_input_fn('data', config['Data']['batch_size'],
                                    config['Monitor']['ckpt_save_epoch'])

    elif dataset_name == 'imagenet':
        return imagenet.input_fn(True, config['Data']['root_path'], config['Data']['batch_size'],
                                 config['Monitor']['ckpt_save_epoch'])


def get_eval_dataset(config):
    dataset_name = config['Data']['name']
    if dataset_name == 'mnist':
        return mnist.eval_input_fn('data', config['Data']['batch_size'])

    elif dataset_name == 'imagenet':
        return imagenet.input_fn(False, config['Data']['root_path'], config['Data']['batch_size'],
                                 1)


def main(_):
    config = load_config(FLAGS.config)
    config['training_graph_path'] = os.path.join(FLAGS.save_path, 'rand_graph')

    dataset_name =  config['Data']['name']
    if dataset_name == 'mnist':
        num_train_images = mnist.MNIST_SIZE
    elif dataset_name == 'imagenet':
        num_train_images = imagenet.NUM_IMAGES['train']
    config['Train']['cos_lr']['step'] = num_train_images * config['Train']['epoch'] / \
                                        config['Data']['batch_size']

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
        run_config = None

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.save_path,
        config=run_config,
        params=config)

    for _ in range(config['Train']['epoch'] // config['Monitor']['ckpt_save_epoch']):
        estimator.train(input_fn=lambda: get_train_dataset(config))
        estimator.evaluate(input_fn=lambda: get_eval_dataset(config))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(main)
