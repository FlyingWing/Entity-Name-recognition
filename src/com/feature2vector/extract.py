#coding:gb2312
'''
Created on 2014��12��25��
@author: Administrator
'''
import os
from com.feature2vector.dir.raw import train_file, test_file
def extracFeature(pos, filename):
    fileTrain = open(train_file, "r")
    file = open(filename, "w")
    str = ""
    for eachline in fileTrain:
        oneline = eachline.strip()
        if oneline == "":
            file.write(str+"\n")
            str = ""
        else:
            lineToken = oneline.split()
            str = str + lineToken[pos]+" " 
    fileTrain.close()
    
    fileTest = open(test_file, "r")
    for eachline in fileTest:
        oneline = eachline.strip()
        if oneline == "":
            file.write(str+"\n")
            str = ""
        else:
            lineToken = oneline.split()
            str = str + lineToken[pos]+" "
    fileTest.close()
    file.close()
    
def writeAll(featureNum):
    for i in xrange(featureNum):
        extracFeature(pos=i+2, filename="feature\\feature_"+str(i)+".txt")
if __name__ == "__main__":
    '''
    ���������˹���ȡ�������Ƚ��٣��˹�ͳ���¾Ϳ����ˣ��ܹ���14������
    '''
    writeAll(14)
    