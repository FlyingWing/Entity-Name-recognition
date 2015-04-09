#coding:gb2312
'''
Created on 2015年1月13日
@author: Administrator

1/27 adopt the token of HLL, modify the BIO to B-NEWGENE,I-NEWGENE
'''
import gzip
import cPickle
import numpy
import theano

from com.liu.dir import raw

def dumpDic():
    print "get dictionary index..."
    file = gzip.open(raw.tag_map_vector_dictioary, "rb")
    dicEmb = cPickle.load(file)
    index = 0
    dic = {}
    word2idx = {}
    embedding = []
    '''
    in the embedding dictionary Line numbers corresponding to the index,
    at last the list will be converted to array
    '''
    for k,v in dicEmb.iteritems():
        word2idx[k] = index
        embedding.append(v)
        index += 1
        '''
        in order to test, only load a little part of dictionary here,
        when run the whole dictionary, pls note the following program
        '''
#        if index > 100: break
    file.close()
    length = len(embedding[index-1])
    print "word emmbeding is ", length
    word2idx["unk"] = index
    embedding.append([0.5]*length)
    
    dic["word2idx"] = word2idx
    print dic["word2idx"]
    dic["embedding2idx"] = numpy.asarray(embedding, dtype = theano.config.floatX)
#    dic["label2idx"] = {"B":0, "I":1, "O":2}
    dic["label2idx"] = {"B-NEWGENE":0, "I-NEWGENE":1, "O":2}
    return dic

def dumpTrain(word2idx, label2idx):
    print "get train index ..."
    file = open(raw.train_raw_file, "r")
    train_x = []
    train_label = []
    sentence = []
    label = []
    for eachline in file:
        oneline = eachline.strip()
        if oneline.startswith("BC2GM"):
            continue
        elif oneline == "":
            train_x.append(numpy.asarray(sentence, dtype = "int32"))
            train_label.append(numpy.asarray(label, dtype = "int32"))
            sentence = []
            label = []
        else:
            token = oneline.split()
            key = token[0].lower();
            if key not in word2idx.keys():
                sentence.append(word2idx["unk"])
            else:
                sentence.append(word2idx[key])
            label.append(label2idx[token[1]])
    file.close()
    return (train_x, train_label)
            
def dumpTest(word2idx):
    '''
    the difference between test_x and test_idx is that
    type(test_idx) is dictionary and it has the sentence id.
    it is used to prediction 
    '''
    print "get test index ..."
    file = open(raw.test_raw_file, "r")
    test_x = []
    test_idx = {}
    sentence = []
    sentenceID = None
    for eachline in file:
        oneline = eachline.strip()
        if oneline.startswith("BC2GM"):
            sentenceID = oneline
            continue
        elif oneline == "":
            test_x.append(numpy.asarray(sentence, dtype = "int32"))
            test_idx[sentenceID] = numpy.asarray(sentence, dtype = "int32")
            sentence = []
        else:
            token = oneline.split()
            key = token[0].lower()
            if key not in word2idx.keys():
                sentence.append(word2idx["unk"])
            else: 
                sentence.append(word2idx[key])
    file.close()
    return (test_x, test_idx)

if __name__ == "__main__":
    dic = dumpDic()
    train = dumpTrain(dic["word2idx"], dic["label2idx"])
    test = dumpTest(dic["word2idx"])
    file = gzip.open(raw.bc2gmData, "wb")
    cPickle.dump((train,test,dic), file)
    file.close()
    
    