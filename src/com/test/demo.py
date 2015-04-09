#coding:gb2312
'''
Created on 2015年1月7日
@author: Administrator
'''

#import time
#cur = time.strftime('%Y-%m-%d',time.localtime(time.time()))
#file = open(cur, "w")
#file.write("nihao")
#file.close()

#import numpy
#a = {0:5, 1:6, 2:7}
#d = [map((lambda k,v : (k,v + 2)), k,v) for k,v in a.iteritems()]
#print d

#import numpy
#import theano
#a = [1,23,4]
#b = numpy.asarray(a, dtype=theano.config.floatX)
#print b
#for element in a:
#    print element
    
#import numpy
#file = "a.npy"
#arr = numpy.arange(10)
#a = numpy.save(file, arr)
#b = numpy.load(file)
#print b
#file.seek(0)
#print numpy.load(file)

#import os
#import os.path
#rootdir = "../liu/paramInfor/Elman/myElman-forward"                                   # 指明被遍历的文件夹
#
#for parent,dirnames,filenames in os.walk(rootdir):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
#    for dirname in  dirnames:                       #输出文件夹信息
#        print "parent is:" + parent
#        print  "dirname is" + dirname
#    for filename in filenames:                        #输出文件信息
#        print "parent is:" + parent
#        print "filename is:" + filename
#    print "the full name of the file is:" + os.path.join(parent,filename) 

#str = "nihao"
#if str == "nihao":
#    print "right"
#    print "\\"+"a.txt"
#try:
#    for i in xrange(3):
#        print i
#        for j in xrange(2):            
#            print j
#            if True:
#                print "lkj"
#                raise 
#    print "sucessfule"
#except:
#    print 'nihao'

a = ["a", "d", "c"]
for i, elem in enumerate(a):
    print i, elem

