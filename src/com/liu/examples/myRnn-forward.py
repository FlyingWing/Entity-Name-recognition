
#coding:gb2312
'''
Created on 2015年3月23日
@author: Administrator
'''
import numpy
import time
import sys
import subprocess
import os
import random

from com.liu.data import myload
from com.liu.rnn.myRnn import model
from com.liu.metrics.accuracy import conlleval
from com.liu.utils.tools import shuffle, minibatch, contextwin
from com.liu.predict_format import prefmt
from com.liu.dir import raw
from com.liu.utils.ImproViterbi import NewViterbi

def getStateTransMatrix(train_y):
    '''
    trian_y :: 训练语料中  标记      
    bb :: int, 存放B-B的个数 ; bi :: int, 存放B-I的个数; bo :: int, 存放B-O的个数 
    ib :: int, 存放I-B的个数; ii :: int, 存放I-I的个数; io :: int, 存放I-O的个数
    ob :: int, 存放O-B的个数; oi :: int, 存放O-I的个数; oo :: int, 存放O-O的个数
            初始状态  bstart:: int, B开始的个数; ostart:: int, O开始的个数
    '''
    bb = 0
    bi = 0
    bo = 0
    ib = 0
    ii = 0
    io = 0
    ob = 0
    oi = 0
    oo = 0
    
    bstart = 0
    ostart = 0
    for eachIterm in train_y:
        firstLabel = eachIterm[0]
        if firstLabel == 0:
            bstart += 1
        else:
            ostart += 1
        lastLabel = firstLabel
        for iterm in eachIterm[1:]:
            if lastLabel == 0:
                if iterm == 0:
                    bb += 1
                elif iterm == 1:
                    bi +=1
                else:
                    bo += 1
            elif lastLabel == 1:
                if iterm == 0:
                    ib += 1
                elif iterm ==1:
                    ii += 1
                else:
                    io += 1
            else:
                if iterm == 0:
                    ob +=1
                elif iterm == 1:
                    oi +=1
                else:
                    oo +=1
            lastLabel = iterm
    print "bb = ", bb
    print "ib = ", bi
    print "ob = ", bo
    b = bb + bi + bo + 0.000001
    i = ib + ii + io + 0.000001
    o = ob + oi + oo + 0.000001
    start = bstart + ostart
    transMatrix = [[1.0*bb/b, 1.0*bi/b, 1.0*bo/b],
                   [1.0*ib/i, 1.0*ii/i, 1.0*io/i],
                   [1.0*ob/o, 1.0*oi/o, 1.0*oo/o]]
    initialState = [1.0*bstart/start, 0, 1.0*ostart/start]        
    
    return numpy.matrix(transMatrix), numpy.asarray(initialState)
if __name__ == "__main__":
#    a = [[0, 0, 2, 1],
#         [2, 0, 1, 2]]
#    print getStateTransMatrix(a)
    
    s = {"seed":1234,
         "nhidden":200,
         "emb_dimension":200,
         "win":5,
         "nepochs":24,
         "lr":0.001,
         "bs":5}
       
    print "load the dataset..."
    train_set, test_set, dic = myload.loadBc2gm()
    idx2label = dict((k,v) for v,k in dic['label2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dic['word2idx'].iteritems())
    
    train_x, train_y = train_set
    test_x, test_idx = test_set
    
    print "统计状态转移概率矩阵...."
    transMatrix, intialState = getStateTransMatrix(train_y)
    print "统计完毕"
#    print transMatrix
#    print intialState
    vocsize = len(set(reduce(\
                       lambda x, y: list(x)+list(y),\
                       train_x+test_x)))

    nclasses = len(set(reduce(\
                       lambda x, y: list(x)+list(y),\
                       train_y)))
    
    nsentences = len(train_x)
    
    print "instanciate the model..." 
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    de = s['emb_dimension'],
                    cs = s['win'],
                    emb= dic["embedding2idx"] )
    
#    fileTime = time.strftime("%Y-%m-%d",time.localtime(time.time()))
#    folder = "..\\paramInfor\\MyRnn(he)\\" + fileTime + os.path.basename(__file__).split('.')[0]
#    os.mkdir(folder)
    rnn.load("..\\paramInfor\\MyRnn(he)\\2015-04-05myRnn-forward-win5-re5-57")
#    print "train with early stopping nepochs ..."
#    s['clr'] = s['lr']
#    for e in xrange(s['nepochs']):
#        # shuffle
#        shuffle([train_x, train_y], s['seed'])
#        tic = time.time()
#        for i in xrange(nsentences):
#            cwords = contextwin(train_x[i], s['win'])
#            words  = map(lambda x: numpy.asarray(x).astype('int32'),\
#                         minibatch(cwords, s['bs']))         
#            labels = train_y[i]
#            for word_batch , label_last_word in zip(words, labels):
#                rnn.train(word_batch, label_last_word, s['clr'])
#        
#        eachFolder =  folder + "\\" + str(e)
#        os.mkdir(eachFolder)
#        rnn.save(eachFolder)
        
#            """修改于2015/4/3"""
#            words = numpy.asarray(cwords).astype("int32")
#            labels = train_y[i]
#            rnn.train(words, labels, s["clr"])

#        print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
#        sys.stdout.flush()
    
#    fileTime = time.strftime("%Y-%m-%d",time.localtime(time.time()))
#    folder = "..\\paramInfor\\MyRnn(he)\\" + fileTime + os.path.basename(__file__).split('.')[0]
#    if not os.path.exists(folder): os.mkdir(folder)
#    else: 
#        folder = folder + "-win5" 
#        os.mkdir(folder)
#    rnn.save(folder)
    
    print " test the model, back into the real world : idx -> words"  
    preFile = open("myRnn-bio.txt","w")
    for x in test_x:        
        lst = rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32'))
        testViterbi = NewViterbi(transMatrix, lst[0].tolist(), intialState)
        path = testViterbi.getFianlPath()
        print "path = ",path
        predictions_test=map(lambda x: idx2label[x], path)
        for eachlag in predictions_test:
            preFile.write(eachlag+" ")
        preFile.write("\n")
        break
    preFile.close()  