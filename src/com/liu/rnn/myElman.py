#coding:gb2312
'''
Created on 2015年1月14日
@author: Administrator
'''
import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class model(object):
    
    def __init__(self, nh, nc, de, cs, emb):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        self.embDic = theano.shared(emb)           
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

        # bundle
        self.params = [self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.embDic[idxs].reshape((idxs.shape[0], de*cs))
        y = T.iscalar('y') # label

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, 
                                sequences=x, 
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1,0,:]#work when run train
        p_y_given_x_sentence = s[:,0,:]#work when run classify
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.mean(T.log(p_y_given_x_lastword)[y])
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function( inputs  = [idxs, y, lr],
                                      outputs = nll,
                                      updates = updates )

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
    def load(self,folder):
        print "loading the params in folder..."
        updates = OrderedDict(( param, theano.shared(numpy.load(os.path.join(folder, name + '.npy')))) for param, name in zip( self.params , self.names))
        loadParam = theano.function(inputs = [],updates = updates)
        loadParam()    