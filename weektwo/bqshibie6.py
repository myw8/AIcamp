#coding:utf-8
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import collections
print(tf.__version__)
import os
import random
import time
import logging
import numpy as np
import pickle
import json
slim = tf.contrib.slim

from PIL import Image

logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint10/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './train_images_resize2/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './test_resize/', 'the test dataset dir')

tf.app.flags.DEFINE_boolean('random_flip_up_down', True, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('face_size', 5,
                            "Choose the first `face_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 96, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 31002, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 2000, "the steps to save")

tf.app.flags.DEFINE_string('log_dir', './log4', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 10, 'Number of epoches')
tf.app.flags.DEFINE_boolean('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'train', 'Running mode. One of {"train", "valid", "test"}')
FLAGS = tf.app.flags.FLAGS


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

      Its parts are:
        scope: The scope of the `Block`.
        unit_fn: The ResNet unit function which takes as input a `Tensor` and
          returns another `Tensor` with the output of the ResNet unit.
        args: A list of length equal to the number of units in the `Block`. The list
          contains one (depth, depth_bottleneck, stride) tuple for each unit in the
          block to serve as argument to unit_fn.
      """

def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:
        # kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            #weights_initializer=slim.variance_scaling_initializer(),
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            #activation_fn=tf.nn.relu,
            activation_fn=tf.nn.elu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                # We do not include batch normalization or activation functions in conv1
                # because the first ResNet unit will perform these. Cf. Appendix of [2].
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)
            # This is needed because the pre-activation variant does not have batch
            # normalization or activation functions in the residual unit output. See
            # Appendix of [2].
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            return net, end_points


def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)

def resnet_v2_101(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_101'):
  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      Block(
          'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
      Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, global_pool,
                   include_root_block=True, reuse=reuse, scope=scope)




class DataIterator:
    def __init__(self, data_dir, test=False):

        self.image1_names=[]
        if test==False:

            self.image_names0 = []
            self.image_names1 = []
            self.image_names2 = []
            self.image_names3 = []
            self.image_names4 = []

            main_path = data_dir

            # main_path='./train_images_resize/'
            anger_path = main_path + '0/'
            happiness_path = main_path + '1/'
            neutral_path = main_path + '2/'
            sadness_path = main_path + '3/'
            surprise_path = main_path + '4/'
            # for file in os.listdir(anger_path):
            self.image_names0 += [os.path.join(anger_path, file) for file in os.listdir(anger_path)]
            self.image_names1 += [os.path.join(happiness_path, file) for file in os.listdir(happiness_path)]
            self.image_names2 += [os.path.join(neutral_path, file) for file in os.listdir(neutral_path)]
            self.image_names3 += [os.path.join(sadness_path, file) for file in os.listdir(sadness_path)]
            self.image_names4 += [os.path.join(surprise_path, file) for file in os.listdir(surprise_path)]
            len0 = len(self.image_names0)
            len1 = len(self.image_names1)
            len2 = len(self.image_names2)
            len3 = len(self.image_names3)
            len4 = len(self.image_names4)
            lenMax = max(len0, len1, len2, len3, len4)
            random.shuffle(self.image_names0)
            random.shuffle(self.image_names1)
            random.shuffle(self.image_names2)
            random.shuffle(self.image_names3)
            random.shuffle(self.image_names4)
            self.imageval_names0 = self.image_names0[:int(len0 * 0.1)]
            self.imagetrain_names0 = self.image_names0[int(len0 * 0.1):]
            self.imageval_names1 = self.image_names1[:int(len1 * 0.1)]
            self.imagetrain_names1 = self.image_names1[int(len1 * 0.1):]
            self.imageval_names2 = self.image_names2[:int(len2 * 0.1)]
            self.imagetrain_names2 = self.image_names2[int(len2 * 0.1):]
            self.imageval_names3 = self.image_names3[:int(len3 * 0.1)]
            self.imagetrain_names3 = self.image_names3[int(len3 * 0.1):]
            self.imageval_names4 = self.image_names4[:int(len4 * 0.1)]
            self.imagetrain_names4 = self.image_names4[int(len4 * 0.1):]
            self.imageval_names = self.imageval_names0 + self.imageval_names1 + self.imageval_names2 + self.imageval_names3 + self.imageval_names4  # val
            '''self.train_aug_names0 = self.imagetrain_names0[:(lenMax - len0)]
            self.train_aug_names1 = self.imagetrain_names1[:(lenMax - len1)]
            self.train_aug_names2 = self.imagetrain_names2[:(lenMax - len2)]
            self.train_aug_names3 = self.imagetrain_names3[:(lenMax - len3)]
            self.train_aug_names4 = self.imagetrain_names4[:(lenMax - len4)]'''

            self.imagetrain_names = self.imagetrain_names0 + self.imagetrain_names1 + self.imagetrain_names2 + self.imagetrain_names3 + self.imagetrain_names4
            '''self.imagetrain_names = self.imagetrain_names0 + self.train_aug_names0 + self.imagetrain_names1 + \
                                    self.train_aug_names1 + self.imagetrain_names2 + self.train_aug_names3 + \
                                    self.imagetrain_names3 + self.train_aug_names3 + self.imagetrain_names4 + self.train_aug_names4'''
            random.shuffle(self.imagetrain_names)
            random.shuffle(self.imageval_names)

            self.train_labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.imagetrain_names]
            self.val_labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.imageval_names]
            # self.image_labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image1_names]
            self.train_names = [str(file_name[len(data_dir):].split(os.sep)[1]) for file_name in self.imagetrain_names]
            self.val_names = [str(file_name[len(data_dir):].split(os.sep)[1]) for file_name in self.imageval_names]

            self.image1_names =self.imagetrain_names+self.imageval_names
            #self.image1_names = self.image_names0 + self.image_names1 + self.image_names2 + self.image_names3 + self.image_names4
            random.shuffle(self.image1_names)
            self.image_labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image1_names]
            self.i_names = [str(file_name[len(data_dir):].split(os.sep)[1]) for file_name in self.image1_names]
        else:
            main_test_path = data_dir
            self.image1_names += [os.path.join(main_test_path, file) for file in os.listdir(main_test_path)]
            random.shuffle(self.image1_names)
            self.image_labels = [i for i in range(len(self.image1_names))]
            self.i_names = [str(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image1_names]






    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):  # 数据增强
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images



    def input_pipeline(self, batch_size, num_epochs=None, aug=False, m="all"):  # 数据读取
        if m == "train":
            self.image_names = self.imagetrain_names
            self.labels = self.train_labels
            self.names = self.train_names
        elif m == "val":
            self.image_names = self.imageval_names
            self.labels = self.val_labels
            self.names = self.val_names
        else:
            print(len(self.image1_names))
            self.image_names = self.image1_names
            self.labels = self.image_labels
            self.names = self.i_names

        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        names_tensor = tf.convert_to_tensor(self.names, dtype=tf.string)
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor,names_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        names = input_queue[2]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=3), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch ,names_batch  = tf.train.shuffle_batch([images, labels,names], batch_size=batch_size,
                                                                       capacity=50000,
                                                                       min_after_dequeue=10000,
                                                                       allow_smaller_final_batch=True)

        return image_batch, label_batch,names_batch




def netmodel(keep_prob, images, labels):  # 网络结构2
    conv1 = slim.conv2d(images, 64, [3, 3], 3, padding='SAME', scope='conv1')
    norm1 = slim.batch_norm(conv1, activation_fn=tf.nn.relu, scope='norm1')
    conv2 = slim.conv2d(norm1, 64, [3, 3],  padding='SAME', scope='conv2')
    norm2 = slim.batch_norm(conv2, activation_fn=tf.nn.relu, scope='norm2')
    max_pool_1 = slim.max_pool2d(norm2, [2, 2], [2, 2], padding='SAME')

    conv3 = slim.conv2d(max_pool_1, 128, [3, 3],  padding='SAME', scope='conv3')
    norm3 = slim.batch_norm(conv3, activation_fn=tf.nn.relu, scope='norm3')
    conv4 = slim.conv2d(norm3, 128, [3, 3],  padding='SAME', scope='conv4')
    norm4 = slim.batch_norm(conv4, activation_fn=tf.nn.relu, scope='norm4')
    max_pool_2 = slim.max_pool2d(norm4, [2, 2], [2, 2], padding='SAME')

    conv5 = slim.conv2d(max_pool_2, 256, [3, 3],  padding='SAME', scope='conv5')
    norm5 = slim.batch_norm(conv5, activation_fn=tf.nn.relu, scope='norm5')
    conv6 = slim.conv2d(norm5, 256, [3, 3],  padding='SAME', scope='conv6')
    norm6 = slim.batch_norm(conv6, activation_fn=tf.nn.relu, scope='norm6')
    conv7 = slim.conv2d(norm6, 256, [3, 3],  padding='SAME', scope='conv7')
    norm7 = slim.batch_norm(conv7, activation_fn=tf.nn.relu, scope='norm7')
    max_pool_3 = slim.max_pool2d(norm7, [2, 2], [2, 2], padding='SAME')

    conv8 = slim.conv2d(max_pool_3, 512, [3, 3],  padding='SAME', scope='conv8')
    norm8 = slim.batch_norm(conv8, activation_fn=tf.nn.relu, scope='norm8')
    conv9 = slim.conv2d(norm8, 512, [3, 3],  padding='SAME', scope='conv9')
    norm9 = slim.batch_norm(conv9, activation_fn=tf.nn.relu, scope='norm9')
    conv10 = slim.conv2d(norm9, 512, [3, 3],  padding='SAME', scope='conv10')
    norm10 = slim.batch_norm(conv10, activation_fn=tf.nn.relu, scope='norm10')
    max_pool_4 = slim.max_pool2d(norm10, [2, 2], [2, 2], padding='SAME')

    conv11 = slim.conv2d(max_pool_4, 512, [3, 3],  padding='SAME', scope='conv11')
    norm11 = slim.batch_norm(conv11, activation_fn=tf.nn.relu, scope='norm11')
    conv12 = slim.conv2d(norm11, 512, [3, 3],  padding='SAME', scope='conv12')
    norm12 = slim.batch_norm(conv12, activation_fn=tf.nn.relu, scope='norm12')
    conv13 = slim.conv2d(norm12, 512, [3, 3],  padding='SAME', scope='conv13')
    norm13 = slim.batch_norm(conv13, activation_fn=tf.nn.relu, scope='norm13')
    max_pool_5 = slim.max_pool2d(norm13, [2, 2], [2, 2], padding='SAME')

    flatten = slim.flatten(max_pool_5)
    fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.tanh, scope='fc1')
    logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.face_size, activation_fn=None, scope='fc2')
    return logits


def build_graph(top_k):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 3], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    names = tf.placeholder(dtype=tf.string, shape=[None], name='name_batch')
    #logits = netmodel(keep_prob, images, labels)

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        logits, end_points = resnet_v2_101(images, 5)
    #print(logits)
    logits = slim.flatten(logits)
    sensitive_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    loss = sensitive_loss
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)

    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(sensitive_loss, global_step=global_step)
    probabilities = tf.nn.softmax(logits)

    tf.summary.scalar('loss', sensitive_loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))
    print(predicted_index_top_k)
    return {'images': images,
            'labels': labels,
            'names': names,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


# graph = build_graph(1)

def train():  # 训练
    print('Begin training')
    data_feeder = DataIterator(data_dir=FLAGS.train_data_dir, test=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        train_images, train_labels,train_names = data_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True,
                                                                             m="all")
        test_images, test_labels,test_names = data_feeder.input_pipeline(batch_size=FLAGS.batch_size, m="val")
        graph = build_graph(top_k=1)

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
        start_step = 0

        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        logger.info('===Training Start===')
        try:
            while not coord.should_stop():
                start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])

                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch,
                             graph['keep_prob']: 0.8}
                _, loss_val, train_summary, step = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']],
                    feed_dict=feed_dict)
                train_writer.add_summary(train_summary, step)
                end_time = time.time()

                if step > FLAGS.max_steps:
                    break
                if step % 10 == 1:
                    logger.info("step {0} loss {1}".format(step, loss_val))

                if step % FLAGS.eval_steps == 1:
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['keep_prob']: 1.0}
                    accuracy_test, test_summary = sess.run(
                        [graph['accuracy'], graph['merged_summary_op']],
                        feed_dict=feed_dict)
                    test_writer.add_summary(test_summary, step)
                    logger.info('===============Eval a batch=======================')
                    logger.info('the step {0} test accuracy: {1}'
                                .format(step, accuracy_test))
                    logger.info('===============Eval a batch=======================')
                if step % FLAGS.save_steps == 1:
                    logger.info('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'),
                               global_step=graph['global_step'])
        except tf.errors.OutOfRangeError:
            logger.info('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=graph['global_step'])
        finally:
            coord.request_stop()
        coord.join(threads)


#train()



def validation():  # 验证生成 json文件
    print('validation')
    test_feeder = DataIterator(data_dir=FLAGS.test_data_dir, test=True)
    result = []
    with tf.Session() as sess:
        test_images, test_labels, test_names = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1,
                                                                          m="all")
        graph = build_graph(1)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # initialize test_feeder's inside state

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

        logger.info('===Start validation===')
        try:
            i = 0
            sum = 0

            while not coord.should_stop():
                i += 1
                start_time = time.time()
                test_images_batch, test_labels_batch, test_names_batch = sess.run(
                    [test_images, test_labels, test_names])
                feed_dict = {graph['images']: test_images_batch,

                             graph['names']: test_names_batch,
                             graph['keep_prob']: 1.0}
                batch_names, indices = sess.run([graph['names'], graph['predicted_index_top_k']], feed_dict=feed_dict)

                leng = len(batch_names)
                sum = sum + leng
                for j in range(leng):
                    temp_dict = {}
                    temp_dict['filename'] = (batch_names[j].decode())
                    temp_dict['label'] = int(indices[j])
                    result.append(temp_dict)

                end_time = time.time()
                logger.info("the batch {0} takes {1} seconds"
                            .format(i, end_time - start_time))

        except tf.errors.OutOfRangeError:
            logger.info('==================Validation Finished================')
        finally:
            coord.request_stop()
        coord.join(threads)

    result_file = 'week2_submission6.json'
    logger.info('Write result into {0}'.format(result_file))

    with open(result_file, 'w') as f:
        json.dump(result, f)
    logger.info('write result json, num is %d' % len(result))


validation()
