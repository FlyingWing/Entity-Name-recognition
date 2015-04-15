#coding:gb2312
'''
Created on 2015年4月11日
@author: Administrator
'''
def getStateTransMatrix(train_y):
    '''
    trian_y :: 训练语料中  标记      
    bb :: int, 存放B-B的个数 ; bi :: int, 存放B-I的个数; bo :: int, 存放B-O的个数 
    ib :: int, 存放I-B的个数; ii :: int, 存放I-I的个数; io :: int, 存放I-O的个数
    ob :: int, 存放O-B的个数; oi :: int, 存放O-I的个数; oo :: int, 存放O-O的个数
            初始状态  bstart:: int, B开始的个数; ostart:: int, O开始的个数
    '''
    bb = 0
    bi = 0
    bo = 0
    ib = 0
    ii = 0
    io = 0
    ob = 0
    oi = 0
    oo = 0
    
    bstart = 0
    ostart = 0
    for eachIterm in train_y:
        firstLabel = eachIterm[0]
        if firstLabel == 0:
            bstart += 1
        else:
            ostart += 1
        lastLabel = firstLabel
        for iterm in eachIterm[1:]:
            if lastLabel == 0:
                if iterm == 0:
                    bb += 1
                elif iterm == 1:
                    bi +=1
                else:
                    bo += 1
            elif lastLabel == 1:
                if iterm == 0:
                    ib += 1
                elif iterm ==1:
                    ii += 1
                else:
                    io += 1
            else:
                if iterm == 0:
                    ob +=1
                elif iterm == 1:
                    oi +=1
                else:
                    oo +=1
            lastLabel = iterm
    print "bb = ", bb
    print "ib = ", bi
    print "ob = ", bo
    b = bb + bi + bo + 0.000001
    i = ib + ii + io + 0.000001
    o = ob + oi + oo + 0.000001
    start = bstart + ostart
    transMatrix = [[1.0*bb/b, 1.0*bi/b, 1.0*bo/b],
                   [1.0*ib/i, 1.0*ii/i, 1.0*io/i],
                   [1.0*ob/o, 1.0*oi/o, 1.0*oo/o]]
    initialState = [1.0*bstart/start, 0, 1.0*ostart/start]        
    
    return numpy.matrix(transMatrix), numpy.asarray(initialState)