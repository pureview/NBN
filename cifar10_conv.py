'''test on mnist dataset'''
import tensorflow as tf
import os
import sys
import shutil
import random
import numpy as np
import cv2
from lib import log_likelihood
from IPython import embed

'''
This NBN implemented with cifar10 dataset.
'''

batchsize = 16
log_dir = './cifar_conv_models'
max_iter = 10000
max_iter_random=500000
training_num=50000
testing_num=10000
momentum = 0.9
learning_rate = 0.00001
save_interval = 1000
print_interval = 10
random_thresh=0.4
label_enlarge=10
random_random=False
start_over = False # if True, delete models dir at first
train_random= False # randomly choose some
# get minist dataset from tensorflow examples
cifar10=tf.contrib.keras.datasets.cifar10.load_data()
image_size=32
image_chanel=3
label_size=10
conv_flag=True
weight_decay=0.5
label_flag=True
cover_thresh=1

class Dataset:
    def __init__(self,conv=True,include_label=True):
        self.conv=conv
        self.train_ind=0
        self.test_ind=0
        self.epoch=0
        self.include_label=include_label
    def next_batch(self,train=True):
        if train:
            # get train batch
            if self.conv:
                # convolution batch
                if self.train_ind+batchsize>training_num:
                    self.epoch+=1
                    ret=np.concatenate([cifar10[0][0][self.train_ind:],cifar10[0][0][:self.train_ind+batchsize-training_num]],axis=0)
                    retlabel=np.concatenate([cifar10[0][1][self.train_ind:],cifar10[0][1][:self.train_ind+batchsize-training_num]],axis=0)
                    self.train_ind=self.train_ind+batchsize-training_num
                    assert ret.shape[0]==batchsize,'batchsize error, ret.shape: '+str(ret.shape)

                else:
                    ret=cifar10[0][0][self.train_ind:self.train_ind+batchsize]
                    retlabel=cifar10[0][1][self.train_ind:self.train_ind+batchsize]
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
                if self.train_ind+batchsize>training_num:
                    self.epoch+=1
                    ret=np.concatenate([cifar10[0][0][self.train_ind:],cifar10[0][0][:self.train_ind+batchsize-training_num]],axis=0)
                    retlabel=np.concatenate([cifar10[0][1][self.train_ind:],cifar10[0][1][:self.train_ind+batchsize-training_num]],axis=0)
                    self.train_ind=self.train_ind+batchsize-training_num
                else:
                    ret=cifar10[0][0][self.train_ind:self.train_ind+batchsize]
                    retlabel=cifar10[0][1][self.train_ind:self.train_ind+batchsize]
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
                if self.test_ind+batchsize>testing_num:
                    self.epoch+=1
                    ret=np.concatenate([cifar10[0][0][self.test_ind:],cifar10[0][0][:self.test_ind+batchsize-testing_num]],axis=0)
                    retlabel=np.concatenate([cifar10[0][1][self.test_ind:],cifar10[0][1][:self.test_ind+batchsize-testing_num]],axis=0)
                    self.test_ind=self.test_ind+batchsize-testing_num
                    assert ret.shape[0]==batchsize,'batchsize error, ret.shape: '+str(ret.shape)

                else:
                    ret=cifar10[0][0][self.test_ind:self.test_ind+batchsize]
                    retlabel=cifar10[0][1][self.test_ind:self.test_ind+batchsize]
                    self.test_ind+=batchsize
                if self.include_label:
                    # include label with ret
                    bone=np.zeros((batchsize,image_size,image_size,label_size))
                    for i in range(batchsize):
                        bone[i,0,0,retlabel[i]]=1
                    #retlabel=np.zeros((batchsize,image_size,image_size,label_size))*bone
                    ret=np.concatenate([ret,bone],axis=-1)
                return ret
            else:
                # fully connected
                if self.test_ind+batchsize>testing_num:
                    self.epoch+=1
                    ret=np.concatenate([cifar10[0][0][self.test_ind:],cifar10[0][0][:self.test_ind+batchsize-testing_num]],axis=0)
                    retlabel=np.concatenate([cifar10[0][1][self.test_ind:],cifar10[0][1][:self.test_ind+batchsize-testing_num]],axis=0)
                    self.test_ind=self.test_ind+batchsize-testing_num
                else:
                    ret=cifar10[0][0][self.test_ind:self.test_ind+batchsize]
                    retlabel=cifar10[0][1][self.test_ind:self.test_ind+batchsize]
                    self.test_ind+=batchsize
                ret=np.reshape(ret,[batchsize,-1])
                if self.include_label:
                    # include label with ret
                    bone=np.zeros((batchsize,label_size))
                    for i in range(batchsize):
                        bone[i,retlabel[i]]=1
                    #retlabel=np.zeros((batchsize,label_size))*bone
                    ret=np.concatenate([ret,bone],axis=1)
                return ret


class Network:
    def __init__(self):
        pass

    def build(self, input,conv=False,pool_size=2,pool_strid=1):
        if not conv:
            # fully connected network, for cifar10, input size is 32*32*3+10=3082
            x=tf.contrib.layers.fully_connected(input,4096,activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            x=tf.contrib.layers.fully_connected(x,8192,activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            x=tf.contrib.layers.fully_connected(x,8192,activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            x=tf.contrib.layers.fully_connected(x,4096,activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            x=tf.contrib.layers.fully_connected(x,image_size*image_size*image_chanel+label_size,activation_fn=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

        else:
            # conv network, for cifar10, input dim is 13
            x=tf.contrib.layers.conv2d(input,16,5,weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            x=tf.contrib.layers.conv2d(x,32,3,weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            #x=tf.contrib.layers.max_pool2d(x,[2,2],stride=1,padding='SAME')
            #x=tf.contrib.layers.conv2d(x,64,3,weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            #x=tf.contrib.layers.conv2d(x,128,3,weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            x=tf.contrib.layers.max_pool2d(x,[2,2],stride=1,padding='SAME')
            x=tf.contrib.layers.conv2d(x,64,3,weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            x=tf.contrib.layers.conv2d(x,32,3,weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            #x=tf.contrib.layers.max_pool2d(x,[2,2],stride=1,padding='SAME')
            #x=tf.contrib.layers.conv2d(x,16,3,weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            x=tf.contrib.layers.conv2d(x,(image_chanel+label_size),3,weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                activation_fn=None)

        tf.summary.histogram('output', x)
        return x

    def loss(self, y, label):
        # here we use l2 loss
        net_loss = tf.nn.l2_loss(y - label)
        regular_loss=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar('net_loss', net_loss)
        tf.summary.scalar('regular_loss', regular_loss)
        return net_loss, regular_loss

def draw_cifar10(x,name='cifar10.jpg'):
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
    pixels=np.zeros([rows*(image_size+2)-2,cols*(image_size+2)-2,image_chanel])
    for i in range(batchsize):
        candidate=np.reshape(x[i,:,:,:image_chanel],[image_size,image_size,image_chanel])
        pixels[int(i/cols)*(image_size+2):int(i/cols)*(image_size+2)+image_size,(i%cols)*(image_size+2):(i%cols)*(image_size+2)+image_size,:]=candidate
    flag=cv2.imwrite(name,pixels)
    if flag:
        pass
        #print('done dump',name,'successful.')
    else:
        print('done dump',name,'fail!')


def cifar10_cls_test(sess,x,y,dataset):
    ''' Using generative model to classify mnist
    Args:
        sess: tensorflow sess
        x: network input placeholder
        y: network output tensor
    '''
    print('************* cls test ******************')
    test_iters=int(testing_num/batchsize)
    total_acc=0
    for i in range(test_iters):
        batch=dataset.next_batch(train=False)
        batchX=batch.copy()
        if conv_flag:
            batchX[:,:,:,-label_size:]=0
        else:
            batchX[:,-10:]=0
        yval=sess.run(y,feed_dict={x:batchX})
        # cal acc
        if conv_flag:
            pred_label=np.sum(np.argmax(np.sum(yval[:,:,:,-label_size:],axis=[1,2]),axis=-1)==np.argmax(batch[:,0,0,-10:],axis=-1))/batchsize
        else:
            total_acc+=np.sum(np.argmax(yval[:,-10:],axis=-1)==np.argmax(batch[:,-10:],axis=-1))/batchsize
    print('total test samples:',test_iters*batchsize)
    print('average acc:',total_acc/test_iters)


def cifar10_gen_cover(sess,x,y,dataset,draw_batch=50,random_thresh=cover_thresh,random_flag=True,square=False,cross=False,
                    square_width=0.3,save_dir='cifar10/',draw_flag=True):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    #test_iters=int(testing_num/batchsize)
    test_iters=draw_batch
    cls_acc=0
    for iter in range(test_iters):
        batch=dataset.next_batch(train=False)
        if draw_flag:
            draw_cifar10(batch,name=save_dir+str(iter)+'-raw.jpg')
        # generate random index
        batchX=batch.copy()
        if conv_flag:
            batchX[:,:,:,-10:]=0
        else:
            batchX[:,-10:]=0
        if square:
            batchX=np.reshape(batchX,[batchsize,28])
        elif cross:
            raise NotImplementedError('I haven\' implement cross cover method')
        elif random_flag:
            # randomly cover input pixels
            for i in range(batchsize):
                ind=np.random.randint(0,image_size*image_size*image_chanel,int(image_size*image_size*image_chanel*random_thresh))
                if conv_flag:
                    batchX[i,ind//image_chanel//image_size,ind//image_chanel%image_size,ind%image_chanel]=0
                    batchX[i,:,:,-10:]=0
                else:
                    batchX[i,ind]=0
                    batchX[i,-10:]=0
        if draw_flag:
            draw_cifar10(batchX,name=save_dir+str(iter)+'-blured.jpg')
        yval = sess.run(y, feed_dict={x: batchX})
        if draw_flag:
            draw_cifar10(yval,name=save_dir+str(iter)+'-predict.jpg')
        # cal acc
        if conv_flag:
            cls_acc+=0
            #cls_acc+=np.sum(np.argmax(np.sum(yval[:,:,:,-label_size:],axis=[1,2]),axis=-1)==np.argmax(batch[:,0,0,-10:],axis=-1))/batchsize
        else:
            cls_acc+=np.sum(np.argmax(yval[:,-10:],axis=-1)==np.argmax(batch[:,-10:],axis=-1))/batchsize
        sys.stdout.write('\r Having finished {0:>5.2%} %'.format(iter / test_iters))
    print('\nthresh=',random_thresh,'cover_acc=',cls_acc/batchsize/test_iters)

def train(network,dataset):
    # conv
    if conv_flag:
        input = tf.placeholder(tf.float32, (None, image_size,image_size,image_chanel+label_size))
        label = tf.placeholder(tf.float32, (None, image_size,image_size,image_chanel+label_size))
    else:
        input = tf.placeholder(tf.float32, (None, image_size*image_size*image_chanel+label_size))
        label = tf.placeholder(tf.float32, (None, image_size*image_size*image_chanel+label_size))
    y = network.build(input,conv=conv_flag)
    #loss, netloss, debugloss = network.softmax_loss(y, label, condition=True)
    net_loss,regular_loss= network.loss(y, label)
    minimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    #minimizer=tf.train.RMSPropOptimizer(learning_rate,momentum=momentum)
    train_op = minimizer.minimize(net_loss+regular_loss)
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
    sess = tf.Session()
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    ckpoint = tf.train.latest_checkpoint(log_dir)
    if ckpoint:
        print('restoring from', ckpoint, '...')
        i = int(ckpoint.split('-')[-1])
        i=0
        saver.restore(sess, ckpoint)
    else:
        i = 0
        print("start training from scratch")
    if train_random:
        print("*************** Begin training **********************")
        moving_acc=0
        skip_count=0
        for plus in range(max_iter_random):
            batch=dataset.next_batch(train=True)
            batchX=batch.copy()
            if conv_flag:
                # convolution, randomly set some pixels to zero
                for i in range(image_size):
                    for j in range(image_size):
                        if np.random.random()>random_thresh:
                            batchX[:,i,j,:]=0
            else:
                # fully connected network
                for i in range(image_size*image_size*image_chanel):
                    if np.random.random()>random_thresh:
                        batchX[:,i]=0
                # randomly mask label
                if np.random.random()>random_thresh:
                    batchX[:,-10:]=0
            net_loss_val,regular_loss_val, _, summary_val = sess.run(
                [net_loss,regular_loss, train_op, summary_op], feed_dict={input: batchX, label: batch})
            if plus % print_interval == 0:
                print(plus, ' -- net_loss_val=', net_loss_val, 'regular_loss_val=', regular_loss_val)
            if plus % save_interval == 0:
                writer.add_summary(summary_val, plus)
                saver.save(sess, log_dir + '/models', plus)
    #exit()
    # test
    cifar10_gen_cover(sess,input,y,dataset)
    print('***********************  Begin Test ******************************')
    #cifar10_cls_test(sess,input,y,dataset)
    
if __name__ == '__main__':
    if start_over:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    dataset=Dataset(conv=conv_flag,include_label=label_flag)
    network = Network()
    train(network,dataset)
