#coding:gb2312
'''
Created on 2015��4��15��
@author: Administrator
'''
import os
currentDir = os.path.dirname(__file__)
index = currentDir.find("RNN")
projectDir = currentDir[:index]

#ԭʼ�˹���ȡ�������ļ�
train_file = projectDir + "Data\\NER-HeHongLei\\raw\\train"
test_file  = projectDir + "Data\\NER-HeHongLei\\raw\\test"

#����������λ��
featureVector = projectDir + "Data\\feature2vector\\vector\\"
#�˹�����ѧϰ�Ĵ������������ʵ�
dictionary = projectDir + "Data\\feature2vector\\dictionary.pkl.gz"

