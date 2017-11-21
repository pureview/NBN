import tensorflow as tf
import pickle
import math
import random
import numpy as np
import os
import shutil
import nputil
import pgmpy.inference
import pgmpy
import time
import sys
from IPython import embed

############# Args #####################
split=0.5
input_size=16
output_size=16
batchsize=32
datapath='dataset/insurance.dat'
pairdatapath='dataset/insurance_pair.dat'
log_dir = './insurance_models'
max_iter = 1000
max_iter_random=500000
test_iter = 20
momentum = 0.9
learning_rate = 0.00001
save_interval = 1000
print_interval = 10
random_thresh=0.4
random_random=True
train_flag =False  # whether train net or not
start_over = False  # if True, delete models dir at first
train_random= True # randomly choose some
#########################################

class Dataset:
    def __init__(self, split=0.8, batchsize=32):
        self.domains, rawset = pickle.load(open(pairdatapath, 'rb'))
        # use big gen dataset
        gibbs=pickle.load(open(datapath,'rb'))
        #################################################################
        self.input_size = 0
        self.code_book = self.generate_code_book(self.domains)
        self.dataset = self.encode(gibbs,big=True)
        self.deduction_dataset=self.encode_deduction(rawset)
        print('dataset size:',len(self.dataset))
        self.deduction_ind=0
        ###################################################################
        #np.random.shuffle(self.dataset)
        n = len(self.dataset)
        self.skip_count = 0
        self.trainset = self.dataset[:int(n * split)]
        self.testset = self.dataset[int(n * split):]
        self.train_ind = 0
        self.test_ind = 0
        self.batch_size = batchsize
        ######################################################################

    def encode_deduction(self,rawset):
        ret = []
        for ob, out in rawset:
            x = [0] * self.input_size
            y = [0.] * self.input_size
            for k in ob:
                x[self.code_book[(k, ob[k])]] = 1
            for k in out:
                y[self.code_book[k]] = out[k]
            ret.append([x, y])
        return np.array(ret)

    def next_batch(self, train=True):
        ''' clean param only useful in training'''
        if train:
            ret = []
            for i in range(self.batch_size):
                if self.train_ind >= self.trainset.shape[0]:
                    # reach the end start from start
                    print('reach the end, shuffle...')
                    np.random.shuffle(self.trainset)
                    self.train_ind = 0
                # clean data
                ret.append(self.trainset[self.train_ind])
                self.train_ind += 1
            return np.stack(ret)
        else:
            # get test data
            ret = []
            for i in range(self.batch_size):
                if self.test_ind >= self.testset.shape[0]:
                    # reach the end start from start
                    print('reach the end, shuffle...')
                    np.random.shuffle(self.testset)
                    self.test_ind = 0
                # clean data
                ret.append(self.testset[self.test_ind])
                self.test_ind += 1
            return np.stack(ret)

    def next_batch_deduction(self,clean=True,onepass=False):
        ''' clean param only useful in training'''

        def is_clean(label):
            for group in self.group_info:
                cursum = 0
                for ind in group:
                    cursum += label[ind]
                if cursum < 0.9:
                    self.skip_count += 1
                    if (self.skip_count + 1) % 100 == 0:
                        print('haven skip', self.skip_count, 'data,current label:', label[group[0]:group[-1] + 1])
                    return False
            return True

        ret = []
        for i in range(self.batch_size):
            while True:
                if self.deduction_ind >= self.deduction_dataset.shape[0]:
                    # reach the end start from start
                    if onepass:
                        return []
                    print('reach the end, shuffle...')
                    np.random.shuffle(self.deduction_dataset)
                    self.deduction_ind = 0
                if clean:
                    if is_clean(self.deduction_dataset[self.deduction_ind, 1]):
                        # clean data
                        ret.append(self.deduction_dataset[self.deduction_ind])
                        self.deduction_ind += 1
                        break
                    else:
                        self.deduction_ind += 1
                else:
                    ret.append(self.deduction_dataset[self.deduction_ind])
                    break
        return np.stack(ret)

    def encode(self, raw,big=False):
        ret = []
        print('find', len(raw), 'pieces of data')
        if big:
            for ind,ob in enumerate(raw):
                #ob=raw[ind]
                x=[0]*self.input_size
                for key in ob:
                    x[self.code_book[(key,self.domains[key][ob[key]])]]=1
                ret.append(x)
            return np.array(ret)
        else:
            for ob in raw:
                x = [0] * self.input_size
                for k in ob[0]:
                    x[self.code_book[(k, ob[0][k])]] = 1
                ret.append(x)
            return np.array(ret)

    def generate_code_book(self, data_dict):
        ''' Generate code book for data and group information'''
        self.group_info = []
        decode_book = []
        code_book = {}
        count = 0
        sk = sorted(data_dict.keys())
        for key in sk:
            vals = data_dict[key]
            temp = []
            for val in vals:
                code_book[(key, val)] = count
                decode_book.append((key, val))
                temp.append(count)
                count += 1
            self.group_info.append(temp)
        self.input_size = count
        self.decode_book = decode_book
        return code_book

    def decode(self, data, label=False):
        ''' data: array'''
        decoded = {}
        if not label:
            for i, d in enumerate(data):
                if d == 1:
                    decoded[self.decode_book[i][0]] = self.decode_book[i][1]
            return decoded
        else:
            for i, d in enumerate(data):
                decoded[self.decode_book[i]] = d
            return decoded


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
        assert input.shape[1] == self.input_size
        if not conv:
            # first layer
            x = self.dense(input, self.input_size, 256, 'dense1')
            # second layer
            x = self.dense(x, 256, 512, 'dense2')
            # 3rd layer
            x = self.dense(x, 512, 1024, 'dense3')
            #4rd layer
            x = self.dense(x, 1024, 2048, 'dense4')
            # 5rd layer
            x = self.dense(x, 2048, 1024, 'dense5')
            # 6rd layer
            x = self.dense(x, 1024, 512, 'dense6')
            # output layer
            x = self.dense(x, 512, self.output_size, 'output', activation=None)
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
        netloss = tf.nn.l2_loss(y-label)
        tf.summary.scalar('netloss', netloss)
        #self.total_loss = self.total_loss + netloss
        l1_loss=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss = tf.add_n(l1_loss) + netloss
        return self.total_loss, netloss, netloss / tf.cast(label.shape[0], tf.float32)

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
                #raise NotImplementedError('gen network loss must filter out unclean data')
                debugloss+=tf.reduce_sum(tf.cast(tf.equal(tf.arg_max(p,-1),tf.arg_max(q,-1)),tf.float32))
                netloss+=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=p,logits=q))
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


def train(dataset, network):
    input = tf.placeholder(tf.float32, (batchsize, dataset.input_size))
    label = tf.placeholder(tf.float32, (batchsize, dataset.input_size))
    y = network.build(input)
    loss, netloss, debugloss = network.softmax_loss(y, label, condition=True)
    #loss, netloss, debugloss = network.l2_loss(y, label)
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
            batch=dataset.next_batch()
            batchX=batch.copy()
            #batchY=np.zeros_like(batchX)
            batchY=batch.copy()
            if random_random:
                random_griddle=min(0.2,max(0.8,random.random()))
                random_groups = [group for group in dataset.group_info if random.random() > random_griddle]
            else:
                random_groups=[group for group in dataset.group_info if random.random()>random_thresh]
            if len(random_groups)==0:
                skip_count+=1
                continue
            for group in random_groups:
                batchX[:,group]=0
                #batchY[:,group]=batch[:,group]
            debugloss_val, loss_val, netloss_val, _, summary_val = sess.run(
                [debugloss, loss, netloss, train_op, summary_op], feed_dict={input: batchX, label: batchY})
            moving_acc += debugloss_val/len(dataset.group_info)
            #moving_acc = debugloss_val/len(random_groups)
            if plus % print_interval == 0:
                print(plus, ' -- loss_val=', loss_val, 'netloss_val=', netloss_val, 'moving acc=',
                      moving_acc/(plus+1-skip_count))
            if plus % save_interval == 0:
                writer.add_summary(summary_val, plus)
                saver.save(sess, log_dir + '/models', plus)

    if train_flag:
        moving_acc=0
        for plus in range(max_iter):
            batch = dataset.next_batch()
            local_count = 0
            minicount=0
            for group in dataset.group_info:
                minicount+=1
                cur_pos = (i + plus) * len(dataset.group_info) + local_count
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
                    print(cur_pos, ' -- loss_val=', loss_val, 'netloss_val=', netloss_val, 'moving acc=',moving_acc/(plus*len(dataset.group_info)+minicount))
                if cur_pos % save_interval == 0:
                    writer.add_summary(summary_val, cur_pos)
                    saver.save(sess, log_dir + '/models', cur_pos)
    # test
    print('***********************  Begin Test ******************************')
    #embed()
    if True:
        # deduction test
        import nputil
        out = network.softmax_output_wraper(y)
        #out=y
        accval=0
        thresh_acc=0
        count=0
        while True:
            batch=dataset.next_batch_deduction(onepass=True)
            if len(batch)==0:
                break
            outval=sess.run(out,feed_dict={input:batch[:,0,:]})
            # calculate thresh acc (0.1)
            thresh_acc+=np.sum(np.abs(outval-batch[:,1,:])<0.1)/batchsize/dataset.input_size
            onehot=nputil.max_onehot(outval,dataset.group_info)
            onehot_label=nputil.max_onehot(batch[:,1,:],dataset.group_info)
            cur=nputil.onehot_acc(onehot_label,onehot,dataset.group_info)
            accval+=cur/batchsize
            print('current accuracy:',cur/batchsize)
            count += 1
        print('there is',count,'batches')
        print('deduction acc=',accval/count)
        print('thresh acc (0.1) =',thresh_acc/count)
        print('*************************************************************')
        exit()

    avg_acc = 0
    dataset.test_ind=0
    for i in range(test_iter):
        batch = dataset.next_batch(train=False)
        for group in dataset.group_info:
            batchX = batch.copy()
            batchY = np.zeros_like(batchX)
            # make this group of batchX zero
            batchX[:, group] = 0
            batchY[:, group] = batch[:, group]
            debugloss_val = sess.run(debugloss, feed_dict={input: batchX, label: batchY})
            avg_acc += debugloss_val

    print('total test acc=', avg_acc / test_iter / len(dataset.group_info))
    # training accuracy
    avg_acc = 0
    for i in range(test_iter):
        batch = dataset.next_batch()
        for group in dataset.group_info:
            batchX = batch.copy()
            batchY = np.zeros_like(batchX)
            # make this group of batchX zero
            batchX[:, group] = 0
            batchY[:, group] = batch[:, group]
            debugloss_val = sess.run(debugloss, feed_dict={input: batchX, label: batchY})
            avg_acc+=debugloss_val
    print('total train acc=', avg_acc / test_iter / len(dataset.group_info))
    print('***********************  End Test ******************************')
    # embed()


if __name__ == '__main__':
    if start_over:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    dataset = Dataset()
    network = Network(dataset.input_size, dataset.input_size, dataset.group_info)
    train(dataset, network)
