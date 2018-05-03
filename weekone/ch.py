from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import pickle
import json
slim=tf.contrib.slim

from PIL import Image



logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)
tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size',3755, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 96, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 12002, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 50, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 2000, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './data/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './data/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 10, 'Number of epoches')
tf.app.flags.DEFINE_boolean('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'train', 'Running mode. One of {"train", "valid", "test"}')
FLAGS = tf.app.flags.FLAGS


class DataIterator:
    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        print(truncate_path)

        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):

            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        random.shuffle(self.image_names)
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]
        self.names = [str(file_name[len(data_dir):].split(os.sep)[1]) for file_name in self.image_names]
    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):

        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)

        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        names_tensor = tf.convert_to_tensor(self.names, dtype=tf.string)
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor, names_tensor], num_epochs=num_epochs)

        #self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

        labels = input_queue[1]
        names = input_queue[2]

        print("=======test===========")
        #print(images_tensor)
        #print(names)

        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch,names_batch = tf.train.shuffle_batch([images, labels,names], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)

        return image_batch, label_batch,names_batch

def netmodel(keep_prob,images):
    conv1=slim.conv2d(images,32,[5,5],1,padding='SAME',scope='conv1')
    norm1=slim.batch_norm(conv1, activation_fn=tf.nn.relu, scope='norm1')
    max_pool_1=slim.max_pool2d(norm1,[2,2],[2,2],padding='SAME')
    conv2=slim.conv2d(max_pool_1,48,[3,3],padding='SAME',scope='conv2')
    norm2=slim.batch_norm(conv2,activation_fn=tf.nn.relu, scope='norm2')
    max_pool_2=slim.max_pool2d(norm2,[2,2],[2,2],padding='SAME')
    conv3=slim.conv2d(max_pool_2,64,[3,3],padding='SAME',scope='conv3')
    norm3=slim.batch_norm(conv3,activation_fn=tf.nn.relu, scope='norm3')
    conv4 = slim.conv2d(norm3,96, [3, 3], padding='SAME', scope='conv4')
    norm4= slim.batch_norm(conv4, activation_fn=tf.nn.relu, scope='norm4')
    max_pool_3=slim.max_pool2d(norm4,[2,2],[2,2],padding='SAME')
    conv5 = slim.conv2d(max_pool_3, 128, [3, 3], padding='SAME', scope='conv5')
    norm5 = slim.batch_norm(conv5, activation_fn=tf.nn.relu, scope='norm5')
    conv6 = slim.conv2d(norm5, 256, [3, 3], padding='SAME', scope='conv6')
    norm6 = slim.batch_norm(conv6, activation_fn=tf.nn.relu, scope='norm6')
    max_pool_4 = slim.max_pool2d(norm6, [2, 2], [2, 2], padding='SAME')

    conv7 = slim.conv2d(max_pool_4, 512, [3, 3], padding='SAME', scope='conv7')
    norm7 = slim.batch_norm(conv7, activation_fn=tf.nn.relu, scope='norm7')
    max_pool_5 = slim.max_pool2d(norm7, [2, 2], [2, 2], padding='SAME')

    flatten = slim.flatten(max_pool_5)
    fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.tanh, scope='fc1')
    logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn=None, scope='fc2')
    return  logits

def build_graph(top_k):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    names = tf.placeholder(dtype=tf.string, shape=[None], name='name_batch')
    logits=netmodel(keep_prob,images)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)

    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)
    probabilities = tf.nn.softmax(logits)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'names' :names,
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


def train():
    print('Begin training')
    train_feeder = DataIterator(data_dir='./data/test/')
    test_feeder = DataIterator(data_dir='./data/test/')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        train_images, train_labels,train_names= train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
        test_images, test_labels,train_names = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)
        print(train_names)
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
                logger.info("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                if step > FLAGS.max_steps:
                    break
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



def validation():
    print('validation')
    test_feeder = DataIterator(data_dir='./data/test/')

    final_predict_val = []
    final_predict_index = []
    groundtruth = []
    result = []
    with tf.Session() as sess:
        test_images, test_labels,test_names= test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)
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
            acc_top_1, acc_top_k = 0.0, 0.0
            while not coord.should_stop():
                i += 1
                temp_dict={}
                start_time = time.time()
                test_images_batch, test_labels_batch,test_names_batch = sess.run([test_images, test_labels,test_names])
                feed_dict = {graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['names']: test_names_batch,
                             graph['keep_prob']: 1.0}
                batch_labels, batch_names,probs, indices, acc_1, acc_k = sess.run([graph['labels'],
                                                                       graph['names'],
                                                                       graph['predicted_val_top_k'],
                                                                       graph['predicted_index_top_k'],
                                                                       graph['accuracy'],
                                                                       graph['accuracy_top_k']],
                                                                                  feed_dict=feed_dict)
                final_predict_val += probs.tolist()
                final_predict_index += indices.tolist()
                groundtruth += batch_labels.tolist()
                temp_dict['filename']=batch_names.tolist()
                temp_dict['label']=indices.tolist()
                print(temp_dict)
                result.append(temp_dict)
                acc_top_1 += acc_1
                acc_top_k += acc_k
                end_time = time.time()
                logger.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1) {3}(top_k)"
                            .format(i, end_time - start_time, acc_1, acc_k))


        except tf.errors.OutOfRangeError:
            logger.info('==================Validation Finished================')
            acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size
            acc_top_k = acc_top_k * FLAGS.batch_size / test_feeder.size
            logger.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k))
        finally:
            coord.request_stop()
        coord.join(threads)


    #return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}
    return result


def main(_):
    print(FLAGS.mode)
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == 'validation':
        result = validation()
        result_file = 'result.json'
        logger.info('Write result into {0}'.format(result_file))
        with open(result_file, 'wb') as f:
            json.dump(result, f)
        logger.info('write result json, num is %d' % len(result))
    elif FLAGS.mode == 'inference':
        image_path = './data/test/00190/13320.png'
        final_predict_val, final_predict_index = inference(image_path)
        logger.info('the result info label {0} predict index {1} predict_val {2}'.format(190, final_predict_index,
                                                                                         final_predict_val))

if __name__ == "__main__":
    tf.app.run()
