import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.append('utils')
from lib.utils.nets import *
from lib.utils.datas import *


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# for test
def sample_y(m, n, ind):
    y = np.zeros([m, n])
    for i in range(m):
        y[i, i % 8] = 1
        # y[:,7] = 1
        # y[-1,0] = 1
    return y


def concat(z, y):
    return tf.concat([z, y], 1)


class CGAN_Classifier(object):
    def __init__(self, generator, discriminator, classifier, data):
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.data = data

        # data
        self.z_dim = self.data.z_dim
        self.y_dim = self.data.y_dim  # condition
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        # nets
        self.G_sample = self.generator(concat(self.z, self.y))

        self.D_real, _ = self.discriminator(self.X)
        self.D_fake, _ = self.discriminator(self.G_sample, reuse=True)

        self.C_real = self.classifier(self.X)
        self.C_fake = self.classifier(self.G_sample, reuse=True)

        # loss
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(
            self.D_real))) + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        self.C_real_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.C_real, labels=self.y))  # real label
        self.C_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_fake, labels=self.y))

        # solver
        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss,
                                                                                       var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.G_loss + self.C_fake_loss,
                                                                                       var_list=self.generator.vars)
        self.C_real_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.C_real_loss,
                                                                                            var_list=self.classifier.vars)
        # self.C_fake_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.C_fake_loss, var_list=self.generator.vars)

        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, sample_dir, ckpt_dir='ckpt', training_epoches=1000000, batch_size=32):
        fig_count = 0
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(training_epoches):
            # update D
            for _ in range(1):
                X_b, y_b = self.data(batch_size)
                self.sess.run(
                    self.D_solver,
                    feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim)}
                )
            # update G
            for _ in range(1):
                self.sess.run(
                    self.G_solver,
                    feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)}
                )
            # update C
            for _ in range(1):
                # real label to train C
                self.sess.run(
                    self.C_real_solver,
                    feed_dict={self.X: X_b, self.y: y_b})
            '''
                # fake img label to train G
                self.sess.run(
                    self.C_fake_solver,
                    feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
            '''
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr, C_real_loss_curr = self.sess.run(
                    [self.D_loss, self.C_real_loss],
                    feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                G_loss_curr, C_fake_loss_curr = self.sess.run(
                    [self.G_loss, self.C_fake_loss],
                    feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; C_real_loss: {:.4}; C_fake_loss: {:.4}'.format(epoch,
                                                                                                              C_real_loss_curr,
                                                                                                              C_fake_loss_curr))

                if epoch % 1000 == 0:
                    y_s = sample_y(16, self.y_dim, fig_count % 10)
                    samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})

                    fig = self.data.data2fig(samples)
                    plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count % 10)),
                                bbox_inches='tight')
                    fig_count += 1
                    plt.close(fig)

                if epoch % 2000 == 0:
                    self.saver.save(self.sess, os.path.join(ckpt_dir, "cgan_classifier.ckpt"))

    def test(self, batch_size=32,test_num=10000):
        '''test acc= 0.997896634615'''
        self.sess.run(tf.global_variables_initializer())
        ckpoint = tf.train.latest_checkpoint('ckpt/')
        if not ckpoint:
            print('There is no checkpoint. Exit...')
            exit()
        self.saver.restore(self.sess,ckpoint)
        tf_acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.C_fake,axis=-1),tf.argmax(self.y,axis=-1)),
                                      tf.float32))
        acc=0
        for i in range(int(test_num/batch_size)):
            X_b, y_b = self.data(batch_size,test=True)
            fake_cls=self.sess.run(tf_acc,
                          feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim)}
                          )
            print('iter',i,': current acc=',fake_cls)
            acc+=fake_cls
        print('test acc=',acc/int(test_num/batch_size))

if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # save generated images
    sample_dir = 'Samples/mnist_cgan_classifier'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # param
    generator = G_conv_mnist()
    discriminator = D_conv_mnist()
    classifier = C_conv_mnist()

    data = mnist()

    # run
    cgan_c = CGAN_Classifier(generator, discriminator, classifier, data)
    #cgan_c.train(sample_dir)
    cgan_c.test()
