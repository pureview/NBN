import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import cv2
#import matplotlib as mpl

#mpl.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import os, sys

sys.path.append('utils')
from IPython import embed
from utils.nets import *
from utils.datas import *

'''
@author: zhouhonggang
@content: wgan cifar10,32*32*3
'''
batch_size=batchsize=32
basic_size=32*32*3    
visual_width=32    

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def process_raw_batch(rawbatch, batchsize=32, network_size=28 * 28):
    raise Exception('这个版本不需要处理原始数据')
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

def draw_cifar10(x,filename='cifar10.jpg',batchsize=batchsize):
    # normalize x
    x=np.maximum(np.minimum(x,1),-1)
    x=(x+1)*255/2
    cols=int(np.sqrt(batchsize))
    rows=cols
    if cols*cols!=batchsize:
        rows=cols+1
    if np.rank(x)==2:
        # fully connected network
        x=np.reshape(x[:,:-10],[-1,image_size,image_size,image_chanel])
    else:
        x=x[:,:,:,:image_chanel]
    pixels=np.zeros([rows*(image_size+2)-2,cols*(image_size+2)-2,image_chanel])
    for i in range(batchsize):
        candidate=np.reshape(x[i],[image_size,image_size,image_chanel])
        pixels[int(i/cols)*(image_size+2):int(i/cols)*(image_size+2)+image_size,(i%cols)*(image_size+2):(i%cols)*(image_size+2)+image_size,:]=candidate
    flag=cv2.imwrite(name,pixels)
    if flag:
        pass
        #print('done dump',name,'successful.')
    else:
        print('done dump',name,'fail!')

def visualize(x, filename, batchsize=32):
    x=(x+1)*255/2
    cols = int(np.sqrt(batchsize))
    rows = cols
    if cols * cols != batchsize:
        rows = int(batchsize / cols) + 1
    pixels = np.zeros([rows *( visual_width+2) - 2, cols * (visual_width+2) - 2,3])
    for i in range(batchsize):
        candidate = np.reshape(x[i], [visual_width,visual_width,3])
        pixels[int(i / cols) * (visual_width+2):int(i / cols) * ( visual_width+2) + visual_width, (i % cols) * ( visual_width+2):(i % cols) * ( visual_width+2) + visual_width,:] = candidate
    flag = cv2.imwrite(filename, pixels)
    if flag:
        pass
        #print('done dump', filename, 'successful.')
    else:
        embed()
        print('done dump', filename, 'fail!')


class WGAN():
    def __init__(self, generator, discriminator, data, random_thresh=0.4,basic_size=0,model_dir='ckpt/'):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        self.model_dir=model_dir
        self.basic_size=basic_size
        self.z_dim = basic_size
        self.X_dim = basic_size 

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

        self.random_thresh = random_thresh

    def train(self, sample_folder, training_epoches=1000000, batch_size=32):
        i = 0
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(model_dir)
        if checkpoint != None:
            self.saver.restore(self.sess, checkpoint)
            i = int(int(checkpoint.split('.')[0].split('-')[-1]) / 1000) + 1
            print('restoring i=', i)
            print('restoring from', checkpoint)
        for epoch in range(training_epoches):
            #----------------------------------------------------#
            # Randomly set random thresh
            #self.random_thresh=random.random()
            
            #----------------------------------------------------#
            # update D
            n_d = 100 if epoch < 25 or (epoch + 1) % 500 == 0 else 5
            for _ in range(n_d):
                batch = self.data.next_batch()
                # generate real batchX and batchY
                batchX = batch.copy()
                #random_groups = [group for group in self.group_info if random.random() > self.random_thresh]
                random_groups=random.sample(range(self.basic_size),int(self.random_thresh*self.basic_size))
                for group in random_groups:
                    batchX[:, group] = 0
                self.sess.run(self.clip_D)
                self.sess.run(
                    self.D_solver,
                    feed_dict={self.X: batch, self.z: batchX}
                )
            # update G
            batch = self.data.next_batch()

            batchX = batch.copy()
            random_groups=random.sample(range(self.basic_size),int(self.random_thresh*self.basic_size))
            for group in random_groups:
                batchX[:, group] = 0
            self.sess.run(
                self.G_solver,
                feed_dict={self.z: batchX}
            )

            # print loss. save images.
            if epoch % 100 == 0 or epoch < 100:
                batch = self.data.next_batch()

                # generate real batchX and batchY
                batchX = batch.copy()
                random_groups=random.sample(range(self.basic_size),int(self.random_thresh*self.basic_size))
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
                    self.saver.save(self.sess, self.model_dir, epoch)
                    samples = self.sess.run(self.G_sample, feed_dict={self.z: batchX})
                    visualize(batchX, sample_folder+'/input' + str(i).zfill(3) + '.jpg')
                    visualize(samples, sample_folder+'/output' + str(i).zfill(3) + '.jpg')

                    i += 1

    def test(self, batchsize=32, test_num=10000):
        ''' Void function'''
        raise NotImplementedError
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(self.model_dir)
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
            batch = self.data.next_batch(train=False)


    def mnist_cls_test(self,batchsize=32,test_num=10000):
        ''' @see mnist.mnist_cls_test'''
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(self.model_dir)
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
            batch = self.data.next_batch(False)

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
        checkpoint = tf.train.latest_checkpoint(self.model_dir)
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
            batch = self.data.next_batch()
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
    def __init__(self,basic_size=0):
        self.name = "G_mlp_mnist"
        self.X_dim = basic_size

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            g = tcl.fully_connected(z, 1024, activation_fn=tf.nn.relu,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(z, 2048, activation_fn=tf.nn.relu,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(z, 1024, activation_fn=tf.nn.relu,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(g, self.X_dim, activation_fn=None,
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
            print(tf.shape(x))
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

class Dataset:
    def __init__(self,cifar10,conv=True,include_label=True):
        self.cifar10=cifar10
        self.conv=conv
        self.train_ind=0
        self.test_ind=0
        self.epoch=0
        self.training_num=len(cifar10[0][0])
        self.testing_num=len(cifar10[1][0])
        self.include_label=include_label

    def next_batch(self,train=True):
        if train:
            # get train batch
            if self.conv:
                # convolution batch
                if self.train_ind+batchsize>training_num:
                    self.epoch+=1
                    ret=np.concatenate([self.cifar10[0][0][self.train_ind:],self.cifar10[0][0][:self.train_ind+batchsize-training_num]],axis=0)
                    retlabel=np.concatenate([self.cifar10[0][1][self.train_ind:],self.cifar10[0][1][:self.train_ind+batchsize-training_num]],axis=0)
                    self.train_ind=self.train_ind+batchsize-training_num
                    assert ret.shape[0]==batchsize,'batchsize error, ret.shape: '+str(ret.shape)

                else:
                    ret=self.cifar10[0][0][self.train_ind:self.train_ind+batchsize]
                    retlabel=self.cifar10[0][1][self.train_ind:self.train_ind+batchsize]
                    self.train_ind+=batchsize
                # normalize ret
                ret=ret/255*2-1
                if self.include_label:
                    # include label with ret
                    bone=np.zeros((batchsize,image_size,image_size,label_size))
                    for i in range(batchsize):
                        bone[i,:,:,retlabel[i]]=1
                    #retlabel=np.zeros((batchsize,image_size,image_size,label_size))*bone
                    ret=np.concatenate([ret,bone],axis=-1)
                return ret
            else:
                # fully connected
                if self.train_ind+batchsize>self.training_num:
                    self.epoch+=1
                    ret=np.concatenate([self.cifar10[0][0][self.train_ind:],self.cifar10[0][0][:self.train_ind+batchsize-self.training_num]],axis=0)
                    retlabel=np.concatenate([self.cifar10[0][1][self.train_ind:],self.cifar10[0][1][:self.train_ind+batchsize-self.training_num]],axis=0)
                    self.train_ind=self.train_ind+batchsize-self.training_num
                else:
                    ret=self.cifar10[0][0][self.train_ind:self.train_ind+batchsize]
                    retlabel=self.cifar10[0][1][self.train_ind:self.train_ind+batchsize]
                    self.train_ind+=batchsize
                ret=np.reshape(ret,[batchsize,-1])
                ret=ret/255*2-1
                if self.include_label:
                    # include label with ret
                    bone=np.zeros((batchsize,label_size))
                    for i in range(batchsize):
                        bone[i,retlabel[i]]=1
                    ret=np.concatenate([ret,bone],axis=1)
                return ret
        else:
            # get test batch
            if self.conv:
                # convolution batch
                if self.test_ind+batchsize>self.testing_num:
                    self.epoch+=1
                    ret=np.concatenate([self.cifar10[0][0][self.test_ind:],self.cifar10[0][0][:self.test_ind+batchsize-self.testing_num]],axis=0)
                    retlabel=np.concatenate([self.cifar10[0][1][self.test_ind:],self.cifar10[0][1][:self.test_ind+batchsize-self.testing_num]],axis=0)
                    self.test_ind=self.test_ind+batchsize-self.testing_num
                    assert ret.shape[0]==batchsize,'batchsize error, ret.shape: '+str(ret.shape)

                else:
                    ret=self.cifar10[0][0][self.test_ind:self.test_ind+batchsize]
                    retlabel=self.cifar10[0][1][self.test_ind:self.test_ind+batchsize]
                    self.test_ind+=batchsize
                ret=ret/255*2-1
                if self.include_label:
                    # include label with ret
                    bone=np.zeros((batchsize,image_size,image_size,label_size))
                    for i in range(batchsize):
                        bone[i,:,:,retlabel[i]]=1
                    #retlabel=np.zeros((batchsize,image_size,image_size,label_size))*bone
                    ret=np.concatenate([ret,bone],axis=-1)
                return ret
            else:
                # fully connected
                if self.test_ind+batchsize>self.testing_num:
                    self.epoch+=1
                    ret=np.concatenate([self.cifar10[0][0][self.test_ind:],self.cifar10[0][0][:self.test_ind+batchsize-self.testing_num]],axis=0)
                    retlabel=np.concatenate([self.cifar10[0][1][self.test_ind:],self.cifar10[0][1][:self.test_ind+batchsize-self.testing_num]],axis=0)
                    self.test_ind=self.test_ind+batchsize-self.testing_num
                else:
                    ret=self.cifar10[0][0][self.test_ind:self.test_ind+batchsize]
                    retlabel=self.cifar10[0][1][self.test_ind:self.test_ind+batchsize]
                    self.test_ind+=batchsize
                ret=np.reshape(ret,[batchsize,-1])
                ret=ret/255*2-1

                if self.include_label:
                    # include label with ret
                    bone=np.zeros((batchsize,label_size))
                    for i in range(batchsize):
                        bone[i,retlabel[i]]=1
                    #retlabel=np.zeros((batchsize,label_size))*bone
                    ret=np.concatenate([ret,bone],axis=1)
                return ret


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    sample_folder = 'Samples/wgan_cifar10'
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # param
    generator = my_G_mlp(basic_size=basic_size)
    discriminator = my_D_mlp()

    # get cifar10 data
    raw_data = tf.contrib.keras.datasets.cifar10.load_data()
    data=Dataset(raw_data,include_label=False,conv=False)
    #print('please check cifar10 dataset');embed();exit()

    # run
    model_dir='ckpt/wgan-cifar10/model'
    wgan = WGAN(generator, discriminator, data,basic_size=basic_size,model_dir=model_dir)
    wgan.train(sample_folder)
    #wgan.mnist_cls_test()
    #wgan.mnist_gen_cover()
