#coding:gb2312
'''
Created on 2015��4��13��
@author: Administrator
�����е��˹�������ʾ��onehot
'''
import numpy
import math
import copy
import gzip, cPickle
from com.liu.dir import raw
from com.onehot.statistics.feaNum import staNum
def buildingOnehotDic(file1, file2):
    '''
    onehotDic :: ��ʾ����token���ֵ�
    lst :: list,����Ԫ��������array, ���ÿ��������ǰ�Ķ����Ʊ�ʾ
    lstDic :: list, ����Ԫ��������dictionary, ��ʾÿ��������Ӧ�Ķ����Ʊ�ʾ
    '''
    onehotDic = {}
    lst = []
    lstDic = []
    dimlength = 0
    for iterm in feanum:
        dimlength += int(math.log(iterm, 2))+1
        lst.append(numpy.zeros(int(math.log(iterm, 2))+1))
        lstDic.append({})
    print "����ά�ȣ�", dimlength
    for eachline in file1:
        fealst = []
        oneline = eachline.strip()
        if oneline != "":
            lineToken = oneline.split("\t")
            for iterm, element in enumerate(lineToken[2:-1]):
                if element in lstDic[iterm]:
                    fealst += lstDic[iterm][element].tolist()
                else:
                    lst[iterm] = incrementOne(lst[iterm])#�����ʾ�Ķ���������1
                    lstDic[iterm][element] = copy.copy(lst[iterm])
                    fealst += lst[iterm].tolist()
            onehotDic[lineToken[0]] = fealst
    for eachline in file2:
        fealst = []
        oneline = eachline.strip()
        if oneline != "":
            lineToken = oneline.split("\t")
            for iterm, element in enumerate(lineToken[2:]):
                if element in lstDic[iterm]:
                    fealst += lstDic[iterm][element].tolist()
                else:
                    lst[iterm] = incrementOne(lst[iterm])
                    lstDic[iterm][element] = copy.copy(lst[iterm])
                    fealst += lst[iterm].tolist()
            onehotDic[lineToken[0]] = fealst
    return onehotDic

def incrementOne(arry):
    isOverflow = False
    nTokenOver = 0
    length = len(arry)
    bit = length-1
    while bit>=0:
        bitsum = arry[bit] + nTokenOver
        if bit == length-1:
            bitsum += 1
        if bitsum >= 2:
            if bit == 0:
                isOverflow = True
                arry[bit] = 0
            else:
                bitsum -= 2
                nTokenOver = 1
                arry[bit] = bitsum
        else:
            arry[bit] = bitsum
            break
        bit -= 1
    return arry 
        
if __name__ == "__main__":
    fileTrain = open(raw.featureTrain, "r")
    fileTest  = open(raw.featureTest, "r")
    feanum = staNum(fileTrain, fileTest)#����һ���б�ÿһ�ж�Ӧ������ȡֵ����
    print feanum 
         
    fileTrain.seek(0)
    fileTest.seek(0)
    fileDic = gzip.open(raw.onehotDic, "wb")
    cPickle.dump(buildingOnehotDic(fileTrain, fileTest), fileDic)
#    print buildingOnehotDic(fileTrain, fileTest)    
