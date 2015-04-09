#coding:gb2312
'''
Created on 2015年1月13日
@author: Administrator
'''
import gzip
import cPickle

from os.path import isfile
from com.liu.dir.raw import bc2gmData

def load(filename):
    assert isfile(filename)
    f = gzip.open(filename,'rb')
    return f

def loadBc2gm():
    f = load(bc2gmData)
    train_set, test_set, dicts = cPickle.load(f)
    return train_set, test_set, dicts

if __name__ == "__main__":
    ''' visualize a few sentences '''

    import pdb
    data = loadBc2gm()

    w2ne, w2la = {}, {}
    train, test, dic = data
    
    w2idx, ne2idx, labels2idx = dic['word2idx'], dic['embedding2idx'], dic['label2idx']
    
    idx2w  = dict((v,k) for k,v in w2idx.iteritems())
    idx2la = dict((v,k) for k,v in labels2idx.iteritems())

    test_x  = test
    train_x, train_label = train
    wlength = 35

    for e in ['train']:
        for sw, sl in zip(eval(e+'_x'), eval(e+'_label')):
            print 'WORD'.rjust(wlength), 'LABEL'.rjust(wlength), "EMBEDDING".rjust(wlength)
            for wx, la in zip(sw, sl): print idx2w[wx].rjust(wlength), idx2la[la].rjust(wlength), str(ne2idx[wx]).rjust(wlength)
            print '\n'+'**'*30+'\n'
            pdb.set_trace()