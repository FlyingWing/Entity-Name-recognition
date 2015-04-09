#coding:gb2312
'''
Created on 2015年1月25日
@author: Administrator
'''
import numpy
import time
import sys
import subprocess
import os
import random

from com.liu.data import myload
from com.liu.rnn.myJordan import model
from com.liu.metrics.accuracy import conlleval
from com.liu.utils.tools import shuffle, minibatch, contextwin
from com.liu.predict_format import prefmt
from com.liu.dir import raw


if __name__ == '__main__':

    s = {'lr':0.001,
         'win':7, # number of words in the context window
         'bs':5, # number of backprop through time steps
         'nhidden':200, # number of hidden units
         'seed':1234,
         'emb_dimension':200, # dimension of word embedding
         'nepochs':18}
         
    print "load the data set ..."
    train_set, test_set, dic = myload.loadBc2gm()
    idx2label = dict((k,v) for v,k in dic['label2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dic['word2idx'].iteritems())

    train_x, train_y = train_set
    test_x,  test_idx  = test_set

    vocsize = len(set(reduce(\
                       lambda x, y: list(x)+list(y),\
                       train_x+test_x)))

    print "vocsize = ", vocsize
    nclasses = len(set(reduce(\
                       lambda x, y: list(x)+list(y),\
                       train_y)))
    
    nsentences = len(train_x)

    print "instanciate the model ..."
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    de = s['emb_dimension'],
                    cs = s['win'],
                    emb= dic["embedding2idx"])
    
    rnn.load("..\\paramInfor\\Jordan(he)\\2015-03-25myJordan-forward-win7-re5-6")
    print "train with early stopping nepochs ..."
    for e in xrange(s['nepochs']):
        # shuffle
        shuffle([train_x, train_y], s['seed'])
        tic = time.time()
        for i in xrange(nsentences):
            cwords = contextwin(train_x[i], s['win'])
            words  = map(lambda x: numpy.asarray(x).astype('int32'),\
                         minibatch(cwords, s['bs']))
            labels = train_y[i]
            for word_batch , label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, s['lr'])

        print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
        sys.stdout.flush()
                    
    fileTime = time.strftime("%Y-%m-%d",time.localtime(time.time()))
    folder = "..\\paramInfor\\Jordan(he)\\" + fileTime + os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)
    else: 
        folder = folder + "-6" 
        os.mkdir(folder)
    rnn.save(folder)
    
    print " test the model, back into the real world : idx -> words"
    preFile = open("Jordan-bio.txt","w")
    for x in test_x:        
        predictions_test=map(lambda x: idx2label[x],rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))
        for eachlag in predictions_test:
            preFile.write(eachlag+" ")
        preFile.write("\n")
    preFile.close()    
    
#    '''
#    my token file and prediction
#    '''
#    print " test the model, back into the real world : idx -> words"
#    resultfolder = "..//results//Jordan" + s['corpus']+"//" + str(s['nepochs']) 
#    if not os.path.exists(resultfolder): os.mkdir(resultfolder)
#    filename = os.path.join(resultfolder, fileTime+'.eval')
#    predictions_test = {}
#    for k,x in test_idx.iteritems():        
#        predictions_test[k]=map(lambda x: idx2label[x],rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))
#    prefmt.predict_test_format(raw.originalTestCorpus, predictions_test, filename)
#    prefmt.post_processing(filename, filename+".post")

#    '''
#    HeHongLei's token file and predictions
#    '''
#    resultfolder = "..//results//"+"Jordan"+s['corpus']+"//" + str(s['nepochs']) 
#    if not os.path.exists(resultfolder): os.mkdir(resultfolder)    
#    preFile = open(os.path.join(resultfolder, 'Jordan-bio.txt'),"w")
#    for x in test_x:        
#        predictions_test=map(lambda x: idx2label[x],rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))
#        for eachlag in predictions_test:
#            preFile.write(eachlag+" ")
#        preFile.write("\n")
#        break
#    preFile.close()
    