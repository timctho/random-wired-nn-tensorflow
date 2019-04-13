import tensorflow as tf
from src.dataset import mnist, imagenet


def get_total_train_iters(config):
    num_train_images = 0
    dataset_name = config['Data']['name']
    if dataset_name == 'mnist':
        num_train_images = mnist.MNIST_SIZE
    elif dataset_name == 'imagenet':
        num_train_images = imagenet.NUM_IMAGES['train']
    return num_train_images * config['Train']['epoch'] // config['Data']['batch_size']


def get_total_loss(predictions, labels, params):
    label_smooth = params['Train']['label_smooth']

    if label_smooth != 0.0:
        one_hot_labels = tf.one_hot(labels, params['Data']['num_class'])
        cls_loss = tf.losses.softmax_cross_entropy(
            logits=predictions, onehot_labels=one_hot_labels, label_smoothing=label_smooth)
    else:
        cls_loss = tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=labels)

    l2_reg_loss = tf.add_n(tf.losses.get_regularization_losses()) * params['Model']['weight_decay']
    loss = cls_loss + l2_reg_loss

    tf.summary.scalar('cls_loss', cls_loss)
    tf.summary.scalar('reg_loss', l2_reg_loss)
    return loss, cls_loss, l2_reg_loss


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
