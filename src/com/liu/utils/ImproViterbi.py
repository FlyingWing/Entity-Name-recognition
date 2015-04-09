#coding:gb2312
'''
Created on 2015��4��8��
@author: Administrator
'''
import numpy as np
class NewViterbi(object):
    '''
    stateTransMatrix :: ״̬ת�Ƹ��ʾ���
    observMatrix :: �۲���ʾ���
    initialMatrix :: ��ʼ״̬����
    '''
    def __init__(self, stateTransMatrix, observMatrix, initialMatrix):
        self.A = stateTransMatrix
        self.B = observMatrix
        self.initialPro = initialMatrix*self.B[0]#��ʼ��
    
        #����һ���б�����������ʱ��t״̬Ϊi�����е���·���и������ֵ
        self.maxPro = [self.initialPro]
        self.lastIndex = []#ʱ��t״̬Ϊi�����е���·���и�������·���ĵ�i-1���ڵ�
        self.finalBestState = 0#����·�����յ�
        
    def compute(self):
        length = len(self.B) #��ȡ����ע���еĳ���
        print "���г��� =",length
        for i in xrange(1, length):
            '''ȡ�������е�ÿһ����self.maxPro�е����һ��Ԫ���е�ÿ��Ԫ�����������ֵ'''
            statePro = []
            lastMaxIndex = []#��¼�õ���ǰʱ��״̬iʱ����һ��ʱ��״̬������.
            #���ת�Ƹ��ʾ��������
            columns = self.A.shape[1]
            print "ת�Ƹ��ʾ��������  =",columns
            for eachColumn in xrange(columns):
                column = self.A[:,eachColumn]
                column = column.reshape(3).getA()[0]#����ת��Ϊ���飬����ȡԪ��
#                print "each column = ", column
#                print "maxPro = ", self.maxPro[-1]
                currentState = column * self.maxPro[-1]
#                print "currentState = ", currentState
                maxIndex = np.argmax(currentState)
#                print "lastMaxIndex = ",maxIndex
#                print "column=", eachColumn
#                print "B = ", self.B[i]
#                print currentState[maxIndex] * self.B[i][eachColumn]
                statePro.append(currentState[maxIndex] * self.B[i][eachColumn])
                lastMaxIndex.append(maxIndex)
            self.maxPro.append(np.asanyarray(statePro))
            self.lastIndex.append(lastMaxIndex)
            self.finalBestState = np.argmax(np.asanyarray(statePro))
#            print "*****max=",np.argmax(np.asanyarray(statePro))
            
    def getFianlPath(self):
        self.compute()
        
        path = []
        bestState = self.finalBestState
        path.append(bestState)
        self.lastIndex.reverse()
        print self.lastIndex
        for each in self.lastIndex:
            bestState = each[bestState]
            path.append(bestState)
        path.reverse()
        return path

if __name__ == "__main__":
    A = [[0.5, 0.3, 0.2],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    
    B = [[0.3, 0.3, 0.4],
         [0.5, 0.3, 0.2],
         [0.3, 0.2, 0.3]]
    
    C = [0.2, 0.4, 0.4]
    
    a = np.matrix(A)
    b = np.asanyarray(B)
    c = np.asanyarray(C)

    print "ת�Ƹ��ʾ���=",a
    print "�۲���ʾ���=",b
    print "��ʼ״̬����=",c
    
    print b
    myViterbi = NewViterbi(a, B, c)
    myViterbi.compute()
    print myViterbi.finalBestState
    print myViterbi.getFianlPath()

