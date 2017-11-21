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
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

'''
I finally made it!!!!!!!!!
total test samples: 10000
iter 14000: average acc: 0.9641
iter average acc: 0.9846
#****************************
#cover test: training iter=205000
thresh= 3 cover_acc= 0.5379
thresh= 2 cover_acc= 0.8792


'''

batchsize = 16
log_dir = './mnist_models'
max_iter = 1000
max_iter_random=500000
test_iter = 20
momentum = 0.9
learning_rate = 0.00001
save_interval = 1000
print_interval = 10
random_thresh=0.4
label_enlarge=10
random_random=False
draw_thresh=2
train_flag = False  # whether train net or not
start_over = False  # if True, delete models dir at first
train_random= False # randomly choose some
# get minist dataset from tensorflow examples
mnist=read_data_sets('dataset',one_hot=True)
network_size=28*28*2+10

class Network:
    def __init__(self, input_size, output_size, group_info):
        self.input_size = input_size
        self.output_size = output_size
        self.total_loss = tf.constant(0.)
        assert group_info != None
        self.group_info = group_info

    def weight_variable(self, shape, wd=0.005,inintializer=tf.contrib.layers.xavier_initializer(),
                        regulizer=tf.contrib.layers.l2_regularizer(0.005)):
        initial = tf.truncated_normal(shape, stddev=0.1)
        ret = tf.Variable(initial)
        ret=tf.get_variable(name='weight',shape=shape,dtype=tf.float32,initializer=inintializer,regularizer=regulizer)
        if wd:
            pass
            #self.total_loss += wd * tf.reduce_sum(tf.abs(ret))
        return ret

    def bias_variable(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def conv(self, x, input_c, output_c, name, stride=1, kernel_size=3,
             activation=tf.nn.relu, wd=0.005, padding='SAME', oned=True):
        # Todo: add 2-d conv support
        if oned:
            # conv-1d
            if len(x.shape) == 2:
                x = tf.expand_dims(x, -1)
            with tf.name_scope(name):
                weight = self.weight_variable([kernel_size, input_c, output_c])
                bias = self.bias_variable([output_c])
                if activation!=None:
                    return activation(tf.nn.bias_add(tf.nn.conv1d(x, weight, stride, padding), bias))
                else:
                    return tf.nn.bias_add(tf.nn.conv1d(x, weight, stride, padding), bias)
        else:
            raise NotImplementedError

    def dense(self, x, input_dim, output_dim, name, activation=tf.nn.relu, wd=0.005):
        with tf.variable_scope(name):
            weight = self.weight_variable((input_dim, output_dim), wd)
            bias = self.bias_variable((output_dim,))
            out = tf.matmul(x, weight) + bias
            if activation:
                return activation(out)
            else:
                return out

    def max_pool1d(self,x,input):
        pass

    def build(self, input,conv=False,pool_size=2,pool_strid=1):
        assert input.shape[1] == self.input_size,'input.shape:'+str(input.shape)+' <--> input_size'+str(self.input_size)
        if not conv:
            # first layer
            x = self.dense(input, self.input_size,2048, 'dense1')
            # second layer
            x = self.dense(x, 2048, 4096, 'dense2')
            # 3rd layer
            x = self.dense(x, 4096, 2048, 'dense3')
            #4rd layer
            #x = self.dense(x, 1024, 2048, 'dense4')
            # 5rd layer
            #x = self.dense(x, 2048, 1024, 'dense5')
            # 6rd layer
            #x = self.dense(x, 1024, 512, 'dense6')
            # output layer
            x = self.dense(x, 2048, self.output_size, 'output', activation=None)
        else:
            # conv network
            input=tf.expand_dims(input,axis=-1)
            x=self.conv(input,1,16,'conv1')
            x=self.conv(x,16,32,'conv2')
            #x=tf.layers.max_pooling1d(x,pool_size,pool_strid,padding='same')
            x=self.conv(x,32,64,'conv3')
            x=self.conv(x,64,64,'conv4')
            #x=tf.layers.max_pooling1d(x,pool_size,pool_strid,padding='same')
            x=self.conv(x,64,32,'conv5')
            x=self.conv(x,32,16,'conv6')
            #x=tf.layers.max_pooling1d(x,pool_size,pool_strid,padding='same')
            x=self.conv(x,16,1,'conv7',activation=None)
            x=tf.reshape(x,tf.shape(x)[0:2])

        tf.summary.histogram('output', x)
        return x

    def loss(self, y, label):
        # here we use l2 loss
        net_loss = tf.nn.l2_loss(y - label)
        self.total_loss += net_loss
        tf.summary.scalar('total_loss', self.total_loss)
        tf.summary.scalar('net_loss', net_loss)
        return self.total_loss, net_loss

    def softmax_metric(self, y, label, thresh=0.1):
        acc = tf.constant(0.)
        for group in self.group_info:
            p = tf.slice(label, [0, group[0]], [-1, len(group)])
            q = tf.slice(y, [0, group[0]], [-1, len(group)])
            # calculate loss
            # filter out unclean data
            softq = tf.nn.softmax(q)
            acc += tf.reduce_sum(tf.cast(tf.abs(softq - p) < thresh, tf.float32))
        return acc / tf.cast(tf.size(y), tf.float32)

    def l2_loss(self,y,label):
        netloss = tf.constant(0.)
        debugloss = tf.constant(0.)
        for ind,group in enumerate(self.group_info):
            p = tf.slice(label, [0, group[0]], [-1, len(group)])
            # p=tf.Print(p,[p])
            q = tf.slice(y, [0, group[0]], [-1, len(group)])
            #netloss+=tf.cond(tf.reduce_sum(p)<0.1,lambda :0.,lambda :tf.reduce_sum(tf.square(p-q)))
            if ind<len(self.group_info)-1:
                netloss+=tf.reduce_sum(tf.square(p-q))
            else:
                netloss += label_enlarge*tf.reduce_sum(tf.square(p - q))
            debugloss+=tf.reduce_sum(tf.abs(p-q))
        tf.summary.scalar('netloss', netloss)
        tf.summary.scalar('debugloss', debugloss)
        #self.total_loss = self.total_loss + netloss
        l1_loss=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss = tf.add_n(l1_loss) + netloss
        return self.total_loss, netloss, debugloss / tf.cast(label.shape[0], tf.float32)

    def softmax_loss(self, y, label, condition=False):
        ''' customed softmax with cross entropy
        Args:
            y: net output, [batchsize,output_dim]
            label: ground true label, [batchsize,output_dim]
            condition: whether filter out unclean groups or not
        '''

        def softmax(x):
            ''' x is a list '''
            sumval = tf.constant(0.)
            for i in x:
                sumval += tf.exp(i)
            return [tf.exp(i) / sumval for i in x]

        def cross_entropy(p, q):
            ''' calculate cross entropy'''
            assert len(p) == len(q)
            sumval = tf.constant(0.)
            debugval = tf.constant(0.)
            for i in range(len(p)):
                sumval -= p[i] * tf.log(q[i])
                debugval += tf.abs(p[i] - q[i])
            return sumval, debugval

        def cond(i, debugloss, netloss, p, q):
            return i < p.shape[0]

        def body(i, debugloss, netloss, p, q):
            debugloss += tf.cond(tf.reduce_sum(p[i]) < 1., lambda: 0., lambda:
            tf.cast(tf.equal(tf.argmax(q[i], axis=0), tf.argmax(p[i], axis=0)), tf.float32))
            netloss += tf.cond(tf.reduce_sum(p[i]) < 1, lambda: 0., lambda:
            tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.slice(p, [i, 0], [1, -1]), logits=tf.slice(q, [i, 0], [1, -1]))))
            i += 1
            return [i, debugloss, netloss, p, q]

        netloss = tf.constant(0.)
        debugloss = tf.constant(0.)
        i = tf.constant(0)
        for group in self.group_info:
            p = tf.slice(label, [0, group[0]], [-1, len(group)])
            # p=tf.Print(p,[p])
            q = tf.slice(y, [0, group[0]], [-1, len(group)])
            #q = tf.Print(q, [p, q], 'p and q: ')
            i = 0
            # calculate loss
            # filter out unclean data
            if condition:
                i, debugloss, netloss, p, q = tf.while_loop(cond, body, [i, debugloss, netloss, p, q])
            else:
                raise NotImplementedError('gen network loss must filter out unclean data')
                # netloss+=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=p,logits=q))
        tf.summary.scalar('netloss', netloss)
        tf.summary.scalar('debugloss', debugloss)
        #self.total_loss = self.total_loss + netloss
        l1_loss=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss = tf.add_n(l1_loss) + netloss
        return self.total_loss, netloss, debugloss / tf.cast(label.shape[0], tf.float32)

    def softmax_output_wraper(self,y):
        ''' Get output prob from logits'''
        ret=[]
        for group in self.group_info:
            q = tf.slice(y, [0, group[0]], [-1, len(group)])
            softq = tf.nn.softmax(q)
            ret.append(softq)
        return tf.concat(ret,-1)

    def onehot_output_wraper(self,y):
        ''' Similar as softmax_output_wraper except that
        this function output onehot format'''
        ret = []
        for group in self.group_info:
            q = tf.slice(y, [0, group[0]], [-1, len(group)])
            softq = tf.nn.softmax(q)
            ret.append(tf.where(softq>=tf.reduce_max(softq,-1),tf.ones_like(softq),tf.zeros_like(softq)))
        return tf.concat(ret,-1)

    def metric(self, y, label, thresh=0.1):
        y = tf.Print(y, [y, label], 'y vs label:', 10)
        mask = tf.abs(y - label) < thresh
        return tf.reduce_sum(tf.cast(mask, tf.float32)) / tf.cast(tf.shape(y)[0], tf.float32)

def process_raw_batch(rawbatch):
    # rawbatch: [images,labels]
    # images: 32x784, labels: 32x10
    newbatch=np.zeros([batchsize,network_size],dtype=np.int32)
    for i in range(batchsize):
        for j in range(28*28):
            if rawbatch[0][i,j]>0:
                newbatch[i,2*j+1]=1
            else:
                newbatch[i,2*j]=1
    # add label to newbatch
    newbatch[:,-10:]=rawbatch[1]
    return newbatch

def print_mnist(x,ind=0):
    print('label:',x[1][ind])
    for i in range(28):
        for j in range(28):
            if x[0][ind,i*28+j]>0:
                print(1,end='')
            else:
                print(0,end='')
        print('')

def draw_mnist(x,blur=False,name='mnist.jpg'):
    def norm(x1,x2):
        ''' normalize x1 and x2 to an integer
         between 0 and 1
         Note: x1 and x2 are batched'''
        assert x1.shape==x2.shape
        x1=np.fmin(np.fmax(x1,0),1)
        x2=np.fmin(np.fmax(x2,0),1)
        if blur:
            p1=x1/np.where((x1+x2)==0,1,(x1+x2))
        else:
            p1=np.where(x1>x2,0.,1.)
            ind=np.where(np.logical_and(x1==0,x2==0))
            for i in range(len(ind[0])):
                p1[ind[0][i],ind[1][i]]=0.5
        p1=(p1*255).astype(np.int32)
        return p1

    if x.shape[1]>28*28*2:
        x=x[:,:-10]
    digit_array=norm(x[:,np.arange(0,28*28*2,2)],x[:,np.arange(1,28*28*2,2)])
    cols=int(np.sqrt(batchsize))
    rows=cols
    if cols*cols!=batchsize:
        rows=cols+1
    pixels=np.zeros([rows*30-2,cols*30-2])
    for i in range(batchsize):
        candidate=np.reshape(digit_array[i],[28,28])
        pixels[int(i/cols)*30:int(i/cols)*30+28,(i%cols)*30:(i%cols)*30+28]=candidate
    flag=cv2.imwrite(name,pixels)
    if flag:
        pass
        #print('done dump',name,'successful.')
    else:
        print('done dump',name,'fail!')

def mnist_cls_test(sess,x,y):
    ''' Using generative model to classify mnist
    Args:
        sess: tensorflow sess
        x: network input placeholder
        y: network output tensor
    '''
    print('************* cls test ******************')
    test_iters=int(mnist.test._num_examples/batchsize)
    total_acc=0
    for i in range(test_iters):
        rawbatch=mnist.test.next_batch(batchsize)
        batch=process_raw_batch(rawbatch)
        batchX=batch.copy()
        batchX[:,-10:]=0
        yval=sess.run(y,feed_dict={x:batchX})
        total_acc+=np.sum(np.argmax(yval[:,-10:],axis=-1)==np.argmax(batch[:,-10:],axis=-1))/batchsize
        #embed()
    print('total test samples:',test_iters*batchsize)
    print('average acc:',total_acc/test_iters)

def mnist_gen_test(sess,x,y):
    def print_encoded_mnist(source,ind=0,blur=False,blur_thresh=0.2):
        for i in range(28):
            for j in range(28):
                if not blur:
                    if source[ind,2*(i*28+j)]>=source[ind,2*(i*28+j)+1]:
                        print(0,end='')
                    else:
                        print(1,end='')
                else:
                    if abs(source[ind,2*(i*28+j)]-source[ind,2*(i*28+j)+1])<0.2:
                        print(1,end='')
                    else:
                        print(0,end='')

            print('')

    print('************* gen test ******************')
    test_iters = int(mnist.test._num_examples / batchsize)
    for i in range(test_iters):
        rawbatch = mnist.test.next_batch(batchsize)
        batch = process_raw_batch(rawbatch)
        batchX = batch.copy()
        # randomly set some pixel to zero
        randind=np.random.randint(0,network_size,int(network_size/2))
        batchX[:,randind]=0
        #batchX[:, :-10]=0
        print('true_label=',np.argmax(batch[0,-10:]))
        print_encoded_mnist(batch)
        print('input image:')
        print_encoded_mnist(batchX)
        print('blur region:')
        yval = sess.run(y, feed_dict={x: batchX})
        print('predict_label=',np.argmax(yval[0,-10:]))
        print_encoded_mnist(yval,blur=True)
        print_encoded_mnist(yval)
        embed()

def mnist_gen_iter(sess,x,y,iter_num=10,batch_num=10,save_dir='mnist/',random_thresh=draw_thresh):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cls_acc=0
    total_acc=[0]*iter_num
    for iter in range(batch_num):
        rawbatch=mnist.test.next_batch(batchsize)
        batch=process_raw_batch(rawbatch)
        draw_mnist(batch,name=save_dir+str(iter)+'-raw.jpg')
        # generate random index
        batchX=batch.copy()
        batchX[:,-10:]=0
        randind=np.random.randint(0,network_size,int(network_size*random_thresh))
        batchX[:,randind]=0
        draw_mnist(batchX,name=save_dir+str(iter)+'-blured.jpg')
        for i in range(iter_num):
            # iter several times
            yval = sess.run(y, feed_dict={x: batchX})
            draw_mnist(yval,name=save_dir+str(iter)+'-iter-'+str(i)+'-predict.jpg')
            batchX=yval
            total_acc[i]+=np.sum(np.argmax(yval[:,-10:],axis=-1)==np.argmax(batch[:,-10:],axis=-1))/batchsize
            batchX[:,-10:]=0
    print('total_acc=',[x/batch_num for x in total_acc])

def mnist_likelihood(sess,x,y):
    batch=np.zeros((batchsize,28*28*2+10))
    for i in range(10):
        batch[i,-10+i]=1
    yval=sess.run(y,feed_dict={x:batch})
    # calculate log likelihood
    # #####################
    #       TODO          #
    #######################
    test_num=mnist.test.images.shape[0]
    test_labels=mnist.test.labels
    test_images=mnist.test.images
    ret=0
    for i in range(test_num):
        for j in range(28*28):
            if test_images[i,j]==1:
                ret+=np.log2(min(max(yval[test_labels[i]][j],0.0001),1))
    return ret

def mnist_gen_cover(sess,x,y,random_thresh=0,test_num=10000,
                    random_flag=True,square=False,cross=False,
                    square_width=0.3,save_dir='mnist/',draw_flag=False):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    test_iters=int(test_num/batchsize)
    cls_acc=0
    for iter in range(test_iters):
        rawbatch=mnist.test.next_batch(batchsize)
        batch=process_raw_batch(rawbatch)
        if draw_flag:
            draw_mnist(batch,name='mnist/'+str(iter)+'-raw.jpg')
        # generate random index
        batchX=batch.copy()
        batchX[:,-10:]=0
        if square:
            batchX=np.reshape(batchX,[batchsize,28])
        elif cross:
            raise NotImplementedError('I haven\' implement cross cover method')
        elif random_flag:
            # randomly cover input pixels
            for i in range(batchsize):
                ind=np.random.randint(0,28*28,int(28*28*random_thresh))
                batchX[i,2*ind]=0
                batchX[i,1+2*ind]=0
        if draw_flag:
            draw_mnist(batchX,name='mnist/'+str(iter)+'-blured.jpg')
        yval = sess.run(y, feed_dict={x: batchX})
        if draw_flag:
            draw_mnist(yval,name='mnist/'+str(iter)+'-predict.jpg')
        cls_acc += np.sum(np.argmax(yval[:, -10:], -1) == np.argmax(batch[:, -10:], -1))
        sys.stdout.write('\r Having finished {0:>5.2%} %'.format(iter / test_iters))
    print('\nthresh=',random_thresh,'cover_acc=',cls_acc/batchsize/test_iters)
    # labels=(list(range(0,10))*10)[:batchsize]
    # print('labels:',labels)
    # batchX=np.zeros([batchsize,28*28*2+10])
    # for i in range(batchsize):
    #     batchX[i,-10+labels[i]]=1
    # yval = sess.run(y, feed_dict={x: batchX})
    # draw_mnist(yval,blur=False)

def train(network,group_info):
    input = tf.placeholder(tf.float32, (batchsize, network_size))
    label = tf.placeholder(tf.float32, (batchsize, network_size))
    y = network.build(input)
    #loss, netloss, debugloss = network.softmax_loss(y, label, condition=True)
    loss, netloss, debugloss = network.l2_loss(y, label)
    minimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    #minimizer=tf.train.RMSPropOptimizer(learning_rate,momentum=momentum)
    train_op = minimizer.minimize(loss)
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
        moving_acc=0
        skip_count=0
        for plus in range(max_iter_random):
            rawbatch=mnist.train.next_batch(batchsize)
            batch=process_raw_batch(rawbatch)
            batchX=batch.copy()
            #batchY=np.zeros_like(batchX)
            batchY=batch.copy()
            if random_random:
                random_griddle=min(0.2,max(0.8,random.random()))
                random_groups = [group for group in group_info if random.random() > random_griddle]
            else:
                random_groups=[group for group in group_info if random.random()>random_thresh]
            if len(random_groups)==0:
                skip_count+=1
                continue
            for group in random_groups:
                batchX[:,group]=0
                #batchY[:,group]=batch[:,group]
            debugloss_val, loss_val, netloss_val, _, summary_val = sess.run(
                [debugloss, loss, netloss, train_op, summary_op], feed_dict={input: batchX, label: batchY})
            #moving_acc += debugloss_val/len(random_groups)
            moving_acc = debugloss_val/network_size
            if plus % print_interval == 0:
                print(plus, ' -- loss_val=', loss_val, 'netloss_val=', netloss_val, 'moving acc=',moving_acc)
                      #moving_acc/(plus+1-skip_count))
            if plus % save_interval == 0:
                writer.add_summary(summary_val, plus)
                saver.save(sess, log_dir + '/models', plus)
    #mnist_cls_test(sess,input,y)
    #mnist_gen_test(sess,input,y)
    #mnist_gen_cover(sess,input,y)
    mnist_gen_iter(sess,input,y)
    if train_flag:
        moving_acc=0
        for plus in range(max_iter):
            brawbatch=mnist.train.next_batch(batchsize)
            batch=process_raw_batch(rawbatch)
            local_count = 0
            minicount=0
            for group in group_info:
                minicount+=1
                cur_pos = (i + plus) * len(group_info) + local_count
                local_count += 1
                batchX = batch.copy()
                batchY = np.zeros_like(batchX)
                # make this group of batchX zero
                batchX[:, group] = 0
                batchY[:, group] = batch[:, group]
                debugloss_val, loss_val, netloss_val, _, summary_val = sess.run(
                    [debugloss, loss, netloss, train_op, summary_op], feed_dict={input: batchX, label: batchY})
                moving_acc+=debugloss_val
                if cur_pos % print_interval == 0:
                    print(cur_pos, ' -- loss_val=', loss_val, 'netloss_val=', netloss_val, 'moving acc=',moving_acc/(plus*len(group_info)+minicount))
                if cur_pos % save_interval == 0:
                    writer.add_summary(summary_val, cur_pos)
                    saver.save(sess, log_dir + '/models', cur_pos)
    # test
    print('***********************  Begin Test ******************************')
    #embed()
    if False:
        # deduction test
        import nputil
        out = network.softmax_output_wraper(y)
        #out=y
        accval=0
        thresh_acc=0
        count=0
        while True:
            rawbatch = mnist.train.next_batch(batchsize)
            batch = process_raw_batch(rawbatch)
            if len(batch)==0:
                break
            outval=sess.run(out,feed_dict={input:batch[:,0,:]})
            # calculate thresh acc (0.1)
            thresh_acc+=np.sum(np.abs(outval-batch[:,1,:])<0.1)/batchsize/network_size
            onehot=nputil.max_onehot(outval,group_info)
            onehot_label=nputil.max_onehot(batch[:,1,:],group_info)
            cur=nputil.onehot_acc(onehot_label,onehot,group_info)
            accval+=cur/batchsize
            print('current accuracy:',cur/batchsize)
            count += 1
        print('there is',count,'batches')
        print('deduction acc=',accval/count)
        print('thresh acc (0.1) =',thresh_acc/count)
        print('*************************************************************')
        exit()

if __name__ == '__main__':
    if start_over:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # get group info
    group_info=[]
    for i in range(28*28):
        group_info.append([i*2,i*2+1])
    group_info.append([i for i in range(28*28*2,28*28*2+10)])
    network = Network(28*28*2+10, 28*28*2+10,group_info)
    train(network,group_info)
