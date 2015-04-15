#coding:gb2312
'''
Created on 2014年12月25日
@author: Administrator
'''
import os
import gzip,cPickle
from com.feature2vector.dir.raw import dictionary, train_file, test_file, featureVector
def readAllDic():
    filenamelist = os.listdir(featureVector)
    featurelistDic = {}#store the each feature dictionary,every element is a dictionary
    for filename in filenamelist:
        end = filename.find(".")
        key = int(filename[4:end])
        featurelistDic[key] = convertDic(featureVector+filename)
    return featurelistDic
def convertDic(filename = None):
    if filename == None:
        raise "pls you lost an argument in the funcition convertDic"
    dictionary = {}
    file = open(filename, "r")
    file.readline()
    for eachline in file:
        oneline = eachline.strip()
        token = oneline.split(None,1)
        dictionary[token[0]] = list(token[1].split())
    file.close()
    return  dictionary
def buildDic(filename_train = None, filename_test=None):
    '''
    convert the choosen features to vector
    '''
    featureDic = readAllDic()
    allToken = {}
    print "read train file"
    file = open(filename_train,"r")
    for eachline in file:
        oneline   = eachline.strip()
        if oneline == "":
            continue
        lineToken = oneline.split()
        if lineToken[0] in allToken.keys():
            continue
        tokenVector = [] # define a vector to map a token
        for i in range(2, len(lineToken)-1):
            tokenVector = tokenVector + featureDic[i-2][lineToken[i]]
        allToken[lineToken[0]] = tokenVector
    file.close() 
    
    print "read test file"
    file = open(filename_test, "r")
    for eachline in file:
        oneline = eachline.strip()
        if oneline == "":
            continue
        lineToken = oneline.split()
        if lineToken[0] in allToken.keys():
            continue
        tokenVector = []
        for i in range(2, len(lineToken)):
            tokenVector = tokenVector + featureDic[i-2][lineToken[i]]
        allToken[lineToken[0]] = tokenVector
    file.close()
    return allToken       
if __name__ == "__main__":
   
    dic = buildDic(train_file,test_file)
    ls = (dic["Comparison"])
    print len(ls)
    print ls[0:50]
    print ls[50:100]
    print ls[100:150]
    file = gzip.open(dictionary, "wb")
    cPickle.dump(dic, file)
    file.close()

    