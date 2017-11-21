import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import cv2
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.append('utils')
from IPython import embed
from utils.nets import *
from utils.datas import *

'''
In this file, I implement WGAN which has similar structure as NBN.
total test samples: 9984
average acc: 0.358874198718
random=1, cls acc= 0.336338141026
random=2, cls acc= 0.292167467949
random=3, cls acc= 0.225861378205

'''


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def process_raw_batch(rawbatch, batchsize=32, network_size=28 * 28 * 2 + 10):
    # rawbatch: [images,labels]
    # images: 32x784, labels: 32x10
    newbatch = np.zeros([batchsize, network_size], dtype=np.int32)
    for i in range(batchsize):
        for j in range(28 * 28):
            if rawbatch[0][i, j] > 0:
                newbatch[i, 2 * j + 1] = 1
            else:
                newbatch[i, 2 * j] = 1
    # add label to newbatch
    newbatch[:, -10:] = rawbatch[1]
    return newbatch


def visualize(x, filename, batchsize=32):
    def norm(x1, x2):
        ''' normalize x1 and x2 to an integer
         between 0 and 1
         Note: x1 and x2 are batched'''
        assert x1.shape == x2.shape
        x1 = np.fmin(np.fmax(x1, 0), 1)
        x2 = np.fmin(np.fmax(x2, 0), 1)
        p1 = np.where(x1 > x2, 0., 1.)
        ind = np.where(np.logical_and(x1 == 0, x2 == 0))
        for i in range(len(ind[0])):
            p1[ind[0][i], ind[1][i]] = 0.5
        p1 = (p1 * 255).astype(np.int32)
        return p1

    if x.shape[1] > 28 * 28 * 2:
        x = x[:, :-10]
    digit_array = norm(x[:, np.arange(0, 28 * 28 * 2, 2)], x[:, np.arange(1, 28 * 28 * 2, 2)])
    cols = int(np.sqrt(batchsize))
    rows = cols
    if cols * cols != batchsize:
        rows = int(batchsize / cols) + 1
    pixels = np.zeros([rows * 30 - 2, cols * 30 - 2])
    for i in range(batchsize):
        candidate = np.reshape(digit_array[i], [28, 28])
        pixels[int(i / cols) * 30:int(i / cols) * 30 + 28, (i % cols) * 30:(i % cols) * 30 + 28] = candidate
    flag = cv2.imwrite(filename, pixels)
    if flag:
        pass
        #print('done dump', filename, 'successful.')
    else:
        print('done dump', filename, 'fail!')


class WGAN():
    def __init__(self, generator, discriminator, data, random_thresh=0.4):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        self.z_dim = 28 * 28 * 2 + 10
        self.X_dim = 28 * 28 * 2 + 10

        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        # nets
        self.G_sample = self.generator(self.z)

        self.D_real, _ = self.discriminator(self.X)
        self.D_fake, _ = self.discriminator(self.G_sample, reuse=True)

        # loss
        self.D_loss = - tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)
        self.G_loss = - tf.reduce_mean(self.D_fake)

        self.D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.D_loss,
                                                                               var_list=self.discriminator.vars)
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.G_loss,
                                                                               var_list=self.generator.vars)

        # clip
        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.vars]

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # get group info
        self.group_info = []
        for i in range(28 * 28):
            self.group_info.append([i * 2, i * 2 + 1])
        self.group_info.append([i for i in range(28 * 28 * 2, 28 * 28 * 2 + 10)])
        self.random_thresh = random_thresh

    def train(self, sample_folder, training_epoches=1000000, batch_size=32):
        i = 0
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint('ckpt')
        if checkpoint != None:
            self.saver.restore(self.sess, checkpoint)
            i = int(int(checkpoint.split('.')[0].split('-')[-1]) / 1000) + 1
            print('restoring i=', i)
            print('restoring from', checkpoint)
        for epoch in range(training_epoches):
            # update D
            n_d = 100 if epoch < 25 or (epoch + 1) % 500 == 0 else 5
            for _ in range(n_d):
                rawX, rawY = self.data(batch_size)
                # generate real batchX and batchY
                batch = process_raw_batch([rawX, rawY])
                batchX = batch.copy()
                random_groups = [group for group in self.group_info if random.random() > self.random_thresh]
                for group in random_groups:
                    batchX[:, group] = 0
                self.sess.run(self.clip_D)
                self.sess.run(
                    self.D_solver,
                    feed_dict={self.X: batch, self.z: batchX}
                )
            # update G
            rawX, rawY = self.data(batch_size)
            # generate real batchX and batchY
            batch = process_raw_batch([rawX, rawY])
            batchX = batch.copy()
            random_groups = [group for group in self.group_info if random.random() > self.random_thresh]
            for group in random_groups:
                batchX[:, group] = 0
            self.sess.run(
                self.G_solver,
                feed_dict={self.z: batchX}
            )

            # print loss. save images.
            if epoch % 100 == 0 or epoch < 100:
                rawX, rawY = self.data(batch_size)
                # generate real batchX and batchY
                batch = process_raw_batch([rawX, rawY])
                batchX = batch.copy()
                random_groups = [group for group in self.group_info if random.random() > self.random_thresh]
                for group in random_groups:
                    batchX[:, group] = 0
                D_loss_curr = self.sess.run(
                    self.D_loss,
                    feed_dict={self.X: batch, self.z: batchX})
                G_loss_curr = self.sess.run(
                    self.G_loss,
                    feed_dict={self.z: batchX})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

                if epoch % 1000 == 0:
                    self.saver.save(self.sess, 'ckpt/similar_structure', epoch)
                    samples = self.sess.run(self.G_sample, feed_dict={self.z: batchX})
                    visualize(batchX, 'Samples/mnist_similar_structure/input' + str(i).zfill(3) + '.jpg')
                    visualize(samples, 'Samples/mnist_similar_structure/output' + str(i).zfill(3) + '.jpg')

                    i += 1

    def test(self, batchsize=32, test_num=10000):
        ''' Void function'''
        raise NotImplementedError
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint('ckpt')
        if checkpoint != None:
            self.saver.restore(self.sess, checkpoint)
            i = int(int(checkpoint.split('.')[0].split('-')[-1]) / 1000) + 1
            print('restoring i=', i)
            print('restoring from', checkpoint)
        else:
            print('there is no saved checkpoint, exit...')
            exit()
        test_iters = int(test_num / batchsize)
        for iter in range(test_iters):
            rawX, rawY = self.data(batch_size, test=True)

    def mnist_cls_test(self,batchsize=32,test_num=10000):
        ''' @see mnist.mnist_cls_test'''
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint('ckpt')
        if checkpoint != None:
            self.saver.restore(self.sess, checkpoint)
            i = int(int(checkpoint.split('.')[0].split('-')[-1]) / 1000) + 1
            print('restoring i=', i)
            print('restoring from', checkpoint)
        else:
            print('there is no saved checkpoint, exit...')
            exit()
        test_iters = int(test_num / batchsize)
        print('************* cls test ******************')
        total_acc = 0
        for iter in range(test_iters):
            rawX, rawY = self.data(batchsize, test=True)
            batch=process_raw_batch([rawX,rawY])
            batchX = batch.copy()
            batchX[:, -10:] = 0
            yval = self.sess.run(self.G_sample, feed_dict={self.z: batchX})
            total_acc += np.sum(np.argmax(yval[:, -10:], axis=-1) == np.argmax(batch[:, -10:], axis=-1)) / batchsize
            sys.stdout.write('\r Having finished {0:>5.2%} %'.format(iter/test_iters))
        print('\ntotal test samples:', test_iters * batchsize)
        print('average acc:', total_acc / test_iters)

    def mnist_gen_cover(self, batchsize=32,test_num=10000,random_thresh=2, random_flag=True, square=False, cross=False,
                        save_dir='wgan_cover/',square_width=0.3):
        '''@see: mnist.mnist_cls_test
        '''
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint('ckpt')
        if checkpoint != None:
            self.saver.restore(self.sess, checkpoint)
            i = int(int(checkpoint.split('.')[0].split('-')[-1]) / 1000) + 1
            print('restoring i=', i)
            print('restoring from', checkpoint)
        else:
            print('there is no saved checkpoint, exit...')
            exit()
        test_iters = int(test_num / batchsize)
        print('************* cls test ******************')
        cls_acc=0
        for iter in range(test_iters):
            rawX, rawY = self.data(batchsize, test=True)
            batch=process_raw_batch([rawX,rawY])
            batchX = batch.copy()
            batchX[:, -10:] = 0
            if square:
                batchX = np.reshape(batchX, [batchsize, 28])
            elif cross:
                raise NotImplementedError('I haven\' implement cross cover method')
            elif random_flag:
                # randomly cover input pixels
                for i in range(batchsize):
                    ind = np.random.randint(0, 28 * 28, int(28 * 28 * random_thresh))
                    batchX[i, 2 * ind] = 0
                    batchX[i, 1 + 2 * ind] = 0
                visualize(batchX, save_dir + str(iter) + '-blured.jpg',batchsize)
                yval = self.sess.run(self.G_sample, feed_dict={self.z: batchX})
                visualize(yval, save_dir + str(iter) + '-predict.jpg',batchsize)
                cls_acc+=np.sum(np.argmax(yval[:,-10:],-1)==np.argmax(batch[:,-10:],-1))
                sys.stdout.write('\r Having finished {0:>5.2%} %'.format(iter / test_iters))
        print('\ncls acc=',cls_acc/batchsize/test_iters)

class my_G_mlp(object):
    def __init__(self):
        self.name = "G_mlp_mnist"
        self.X_dim = 28 * 28 * 2 + 10

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            g = tcl.fully_connected(z, 2048, activation_fn=tf.nn.relu,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(z, 4096, activation_fn=tf.nn.relu,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(z, 2048, activation_fn=tf.nn.relu,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(g, self.X_dim, activation_fn=tf.nn.sigmoid,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
        return g

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class my_D_mlp():
    def __init__(self):
        self.name = "D_mlp_mnist"

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu,
                                         weights_initializer=tf.random_normal_initializer(0, 0.02))
            d = tcl.fully_connected(shared, 1, activation_fn=None,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))

            q = tcl.fully_connected(shared, 10, activation_fn=None,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))  # 10 classes

        return d, q

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    sample_folder = 'Samples/mnist_wgan_mlp'
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # param
    generator = my_G_mlp()
    discriminator = D_mlp_mnist()

    data = mnist('mlp')

    # run
    wgan = WGAN(generator, discriminator, data)
    #wgan.mnist_cls_test()
    wgan.mnist_gen_cover()