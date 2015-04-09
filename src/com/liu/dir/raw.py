#coding:gb2312
'''
Created on 2015年1月13日
@author: Administrator
'''
import os
currentDir = os.path.abspath(__file__)
index = currentDir.find("RNN")
projectDir = currentDir[:index]

tag_map_vector_dictioary = projectDir + "Data\\word2vector\\hhl_vectors_s200_Dictionary.pkl.gz"
train_raw_file = projectDir + "Data\\NER-HeHongLei\\raw\\train.raw"
test_raw_file  = projectDir + "Data\\NER-HeHongLei\\raw\\test.raw"
bc2gmData = projectDir + "Data\\NER-HeHongLei\\raw\\bc2gm.pkl.gz"

#tag_map_vector_dictioary = projectDir + "Data\\word2vector\\vectors_s200_Dictionary.pkl.gz"
#train_raw_file = projectDir + "Data\\bc2GMtrain_1.1\\train\\bc2GMtrainRaw.txt"
#test_raw_file  = projectDir + "Data\\bc2GMtest_1.0\\BC2GM\\test\\bc2GMtestRaw.txt"
#bc2gmData = projectDir + "Data\\rnn\\bc2gm.pkl.gz"
originalTestCorpus = projectDir + "Data\\bc2GMtest_1.0\\BC2GM\\test\\test.in"