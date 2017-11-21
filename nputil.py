''' numpy util file'''
import numpy as np
from IPython import embed

def max_onehot(x,group_info):
    ret=[]
    for group in group_info:
        temp=x[:,group]
        maxval=np.amax(temp,-1,keepdims=True)
        temp[temp>=maxval]=1
        temp[temp<maxval]=0
        ret.append(temp)
    return np.concatenate(ret,-1)

def onehot_acc(label,y,group_info):
    ret=0
    for group in group_info:
        label_ind=np.argmax(label[:,group],axis=-1)
        y_ind=np.argmax(y[:,group],axis=-1)
        equal=label_ind==y_ind
        ret+=np.sum(equal.astype(np.float32))
    return ret/len(group_info)

def group_softmax(x,group_info):
    ''' x is raw output of network'''
    ret=[]
    for group in group_info:
        temp=x[:,group]
        Z=np.sum(np.exp(temp),-1,keepdims=True)
        Z=np.repeat(Z,len(group),axis=-1)
        ret.append(np.exp(temp)/Z)
    return np.concatenate(ret,axis=-1)

def KL_divergence(P,Q,group_info):
    ''' P,Q are batched'''
    ret=0.
    for group in group_info:
        tp=P[:,group]
        tq=Q[:,group]
        ret+=np.sum(np.where((tp>0)&(tq>0),tp*np.log(tp/tq),np.zeros_like(tp)))
    return ret/P.shape[0]

def MTA(P,Q,step=0.01):
    ''' P,Q are batched'''
    assert P.shape==Q.shape
    ret = [0]*int(1/step)
    diff=np.abs(P-Q)
    for i in range(int(1/step)):
        thresh=(i+1)*step
        ret[i]=np.sum(diff<thresh)/np.size(P)
    return ret