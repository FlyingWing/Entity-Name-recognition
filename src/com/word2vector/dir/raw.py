#coding:gb2312
'''
Created on 2015年1月13日
@author: Administrator
'''
import os
currentDir = os.path.abspath(__file__)
index = currentDir.find("tokenazition")
projectDir = currentDir[:index]

embeddingTable = projectDir + "Data\\word2vector\\vectors_s200_w7_b0.bin"
train_raw_file = projectDir + "Data\\NER-HeHongLei\\raw\\train.raw"
test_raw_file  = projectDir + "Data\\NER-HeHongLei\\raw\\test.raw"
train_file = projectDir + "Data\\NER-HeHongLei\\raw\\train"
test_file  = projectDir + "Data\\NER-HeHongLei\\raw\\test"

M_biofile  = projectDir + "Data\\NER-HeHongLei\\MEMM-bio.txt"
J_biofile  = projectDir + "Data\\NER-HeHongLei\\Jordan-bio.txt"
E_biofile  = projectDir + "Data\\NER-HeHongLei\\Elman-bio.txt"
My_biofile  = projectDir + "Data\\NER-HeHongLei\\myRnn-bio.txt"
result  = projectDir + "Data\\NER-HeHongLei\\test.ans"
