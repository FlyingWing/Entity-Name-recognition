#coding:gb2312
'''
Created on 2015年4月8日
@author: Administrator
'''
import numpy as np
class NewViterbi(object):
    '''
    stateTransMatrix :: 状态转移概率矩阵
    observMatrix :: 观测概率矩阵
    initialMatrix :: 初始状态概率
    '''
    def __init__(self, stateTransMatrix, observMatrix, initialMatrix):
        self.A = stateTransMatrix
        self.B = observMatrix
        self.initialPro = initialMatrix*self.B[0]#初始化
    
        #定义一个列表用来存贮在时刻t状态为i的所有单个路径中概率最大值
        self.maxPro = [self.initialPro]
        self.lastIndex = []#时刻t状态为i的所有单个路径中概率最大的路径的第i-1个节点
        self.finalBestState = 0#最优路径的终点
        
    def compute(self):
        length = len(self.B) #获取待标注序列的长度
        print "序列长度 =",length
        for i in xrange(1, length):
            '''取出矩阵中的每一列与self.maxPro中的最后一个元素中的每个元素相乘求得最大值'''
            statePro = []
            lastMaxIndex = []#记录得到当前时刻状态i时，上一个时刻状态的索引.
            #获得转移概率矩阵的列数
            columns = self.A.shape[1]
            print "转移概率矩阵的列数  =",columns
            for eachColumn in xrange(columns):
                column = self.A[:,eachColumn]
                column = column.reshape(3).getA()[0]#矩阵转化为数组，便于取元素
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

    print "转移概率矩阵=",a
    print "观测概率矩阵=",b
    print "初始状态概率=",c
    
    print b
    myViterbi = NewViterbi(a, B, c)
    myViterbi.compute()
    print myViterbi.finalBestState
    print myViterbi.getFianlPath()

