#!/usr/bin/python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse

# Enable TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data
# from scipy.misc import imread, imresize
import imageio.v2 as imageio
from skimage.transform import resize
from imagenet_classes import class_names
import numpy as np

# Replace tf.app.flags with argparse
parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train', 
                   help='The phase can be train, eval or test')
parser.add_argument('--load', action='store_true',
                   help='Turn on to load a pretrained model from either the latest checkpoint or a specified file')
parser.add_argument('--model_file', default=None,
                   help='If specified, load a pretrained model from this file')
parser.add_argument('--load_cnn', action='store_true',
                   help='Turn on to load a pretrained CNN model')
parser.add_argument('--cnn_model_file', default='./vgg16_no_fc.npy',
                   help='The file containing a pretrained CNN model')
parser.add_argument('--train_cnn', action='store_true',
                   help='Turn on to train both CNN and RNN. Otherwise, only RNN is trained')
parser.add_argument('--beam_size', type=int, default=3,
                   help='The size of beam search for caption generation')
parser.add_argument('--image_file', default='./man.jpg',
                   help='The file to test the CNN')

FLAGS = parser.parse_args()

## Start token is not required, Stop Tokens are given via "." at the end of each sentence.
## TODO : Early stop functionality by considering validation error. We should first split the validation data.

def main(argv):
    config = Config()
    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size
    config.trainable_variable = FLAGS.train_cnn

    with tf.compat.v1.Session() as sess:
        if FLAGS.phase == 'train':
            # training phase
            data = prepare_train_data(config)
            model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            if FLAGS.load:
                model.load(sess, FLAGS.model_file)
            #load the cnn file
            if FLAGS.load_cnn:
                model.load_cnn(sess, FLAGS.cnn_model_file)
            tf.get_default_graph().finalize()
            model.train(sess, data)

        elif FLAGS.phase == 'eval':
            # evaluation phase
            coco, data, vocabulary = prepare_eval_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval(sess, coco, data, vocabulary)

        elif FLAGS.phase == 'test_loaded_cnn':
            # testing only cnn
            model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
            probs = model.test_cnn(imgs)
            model.load_cnn(sess, FLAGS.cnn_model_file)

            img1 = imageio.imread(FLAGS.image_file)
            img1 = resize(img1, (224, 224), preserve_range=True).astype(np.uint8)

            prob = sess.run(probs, feed_dict={imgs: [img1]})[0]
            preds = (np.argsort(prob)[::-1])[0:5]
            for p in preds:
                print(class_names[p], prob[p])

        else:
            # testing phase
            data, vocabulary = prepare_test_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.test(sess, data, vocabulary)

if __name__ == '__main__':
    main(None)