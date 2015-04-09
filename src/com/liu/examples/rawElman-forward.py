#coding:gb2312
'''
Created on 2015年1月21日
@author: Administrator
'''
import numpy
import time
import sys
import subprocess
import os
import random

from com.liu.data import myload
from com.liu.rnn.elman import model
from com.liu.metrics.accuracy import conlleval
from com.liu.utils.tools import shuffle, minibatch, contextwin
from com.liu.predict_format import prefmt
from com.liu.dir import raw

if __name__ == '__main__':

    s = {"seed":1234,
         "nhidden":200,
         "emb_dimension":200,
         "win":7,
         "nepochs":50,
         "lr":0.0627142536696559,
         "bs":5,
         "verbose":1}

    fileTime = time.strftime("%Y-%m-%d",time.localtime(time.time()))
    folder = "..\\" + fileTime + os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    print "load the dataset..."
    train_set, test_set, dic = myload.loadBc2gm()
    idx2label = dict((k,v) for v,k in dic['label2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dic['word2idx'].iteritems())

    train_x, train_y = train_set
    test_x,  test_idx  = test_set

    vocsize = len(set(reduce(\
                       lambda x, y: list(x)+list(y),\
                       train_x+test_x)))
    print "vocsize=",vocsize
    nclasses = len(set(reduce(\
                       lambda x, y: list(x)+list(y),\
                       train_y)))
    
    nsentences = len(train_x)

    # instanciate the model 
    print "instanciate the model..." 
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    ne = vocsize,
                    de = s['emb_dimension'],
                    cs = s['win'] )

    # train with early stopping on validation set
    print "train with early stopping nepochs ..."
    s['clr'] = s['lr']
    for e in xrange(s['nepochs']):
        # shuffle
        shuffle([train_x, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()
        for i in xrange(nsentences):
            cwords = contextwin(train_x[i], s['win'])
            words  = map(lambda x: numpy.asarray(x).astype('int32'),\
                         minibatch(cwords, s['bs']))
            labels = train_y[i]
            for word_batch , label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, s['clr'])
                rnn.normalize()
            if s['verbose']:
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                sys.stdout.flush()
            
        # evaluation // back into the real world : idx -> words
    print " test the model, back into the real world : idx -> words"
    curTime = fileTime
    filename = "predictions//bc2gm_"+str(s["nepochs"])+"_"+curTime+".eval"

    predictions_test = {}
    for k,x in test_idx.iteritems():        
       predictions_test[k]=map(lambda x: idx2label[x],rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))
    print "start predict..."
    prefmt.predict_test_format(raw.originalTestCorpus, predictions_test, filename)
    prefmt.post_processing(filename, filename+".post")
    rnn.save(folder)
