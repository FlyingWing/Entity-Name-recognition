#coding:gb2312
'''
Created on 2015年4月15日
@author: Administrator
'''
import os
currentDir = os.path.dirname(__file__)
index = currentDir.find("RNN")
projectDir = currentDir[:index]

#原始人工抽取的特征文件
train_file = projectDir + "Data\\NER-HeHongLei\\raw\\train"
test_file  = projectDir + "Data\\NER-HeHongLei\\raw\\test"

#向量特征的位置
featureVector = projectDir + "Data\\feature2vector\\vector\\"
#人工特征学习的词向量，建立词典
dictionary = projectDir + "Data\\feature2vector\\dictionary.pkl.gz"

