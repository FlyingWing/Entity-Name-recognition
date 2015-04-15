#coding:gb2312
'''
@author: Administrator
    把训练到的word2vector映射到token中，提取出对应向量表示，放到一个新的字典中
key:: String
value::list, String
'''
import os
from com.word2vector.dir import raw
class MapVector(object):
    def __init__(self, vectorTable = raw.embeddingTable):
            
        if os.path.exists(vectorTable):
            self.vectorTable = vectorTable
        else:
            raise vectorTable + '\t is not exist'
     
    def getAllTokens(self,train_file_name = raw.train_raw_file,test_file_name  = raw.test_raw_file):
        '''
        get all the tokens in train file and test file and store them in set
        '''
        self.allToken = set()
        trainFile = open(train_file_name, "r")
        for eachline in trainFile:
            oneline = eachline.strip()
            if oneline != "" and not oneline.startswith("BC2GM"):
                token = oneline.split()
                key = token[0].lower()
                self.allToken.add(key)
        trainFile.close()
        testFile = open(test_file_name, "r")
        for eachline in testFile:
            oneline = eachline.strip()
            if oneline != "" and not oneline.startswith("BC2GM"):
                token = oneline.split()
                key = token[0].lower()
                self.allToken.add(key)
        testFile.close()
        self.allToken.add("</s>")
                
    def bulidDic(self, DicName = "vectors_s200_Dictionary.pkl.gz", ndim = 200):
        '''
        map all the tokens in the set to a vector
        DicName : dictionary, key::token, value::list; iterm in list:: string
        ndim    : size of vector (int)
        
        '''
        import gzip, cPickle
        dic = {}
        file = open(self.vectorTable, 'r')
        for eachline in file:
            oneline = eachline.strip()
            alist = oneline.split(None, 1)
            token = alist[0].lower()
            if token in self.allToken:
                dic[token] = alist[1].split()
                self.allToken.remove(token)
        file.close()
        reminder = len(self.allToken)
        if reminder != 0:
            print "%d token in the set" % reminder
            for element in self.allToken:
                dic[element] = ["0"] * ndim
        print "write dictionary..."
        file = gzip.open(DicName, "wb")
        cPickle.dump(dic, file)
    
if __name__ == '__main__':
#    DicSet = MapVector()
#    print "building token set ..."
#    DicSet.getAllTokens()
#    print len(DicSet.allToken)
#    print "budilding dictionary ..."
#    DicSet.bulidDic()

    str = "1 2 3 4"
    dic = {}
    dic['a']=str.split()
#    print dic
    for k, v in dic.iteritems():
        print k
        print v