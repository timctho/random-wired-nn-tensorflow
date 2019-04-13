import tensorflow as tf

from src.model.small_regime import SmallRandWireNN
from src.model.regular_regime import RegularRandWireNN
from src.model.mnist_regime import MnistRandWireNN
from src.model.util import get_total_loss

from src.util.lr_selector import LearningRateSelector
from src.util.opt_selector import OptimizerSelector

import horovod.tensorflow as hvd


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

    loss, cls_loss, l2_reg_loss = get_total_loss(predictions, labels, params)

    summary_op = tf.summary.merge_all()

    train_hooks = [
        tf.train.LoggingTensorHook({'step': global_step,
                                    'lr': lr,
                                    'cls_loss': cls_loss,
                                    'reg_loss': l2_reg_loss},
                                   every_n_iter=params['Monitor']['log_step']),
        tf.train.SummarySaverHook(save_steps=params['Monitor']['log_step'],
                                  summary_op=summary_op,
                                  output_dir=params['save_path'])
    ]

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                          training_chief_hooks=train_hooks)

    elif mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'Top@1': tf.metrics.accuracy(labels, tf.argmax(predictions, axis=-1)),
            'Top@5': tf.metrics.mean(tf.nn.in_top_k(predictions, labels, 5))
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def horovod_model_fn(features, labels, mode, params):
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

    loss, cls_loss, l2_reg_loss = get_total_loss(predictions, labels, params)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'Top@1': tf.metrics.accuracy(labels, tf.argmax(predictions, axis=-1)),
            'Top@5': tf.metrics.mean(tf.nn.in_top_k(predictions, labels, 5))
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    lr = LearningRateSelector(params, global_step).lr * hvd.size()
    opt = OptimizerSelector(params, lr).optimizer
    opt = hvd.DistributedOptimizer(opt)

    summary_op = tf.summary.merge_all()

    train_hooks = [
        hvd.BroadcastGlobalVariablesHook(0),
        tf.train.LoggingTensorHook({'step': global_step,
                                    'lr': lr,
                                    'cls_loss': cls_loss,
                                    'reg_loss': l2_reg_loss},
                                   every_n_iter=params['Monitor']['log_step']),
        tf.train.SummarySaverHook(save_steps=params['Monitor']['log_step'],
                                  summary_op=summary_op,
                                  output_dir=params['save_path'])
    ]

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=train_hooks)


