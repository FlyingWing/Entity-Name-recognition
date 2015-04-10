#coding:gb2312
'''
Created on 2015年1月7日
@author: Administrator
'''
import operator
x = {"a":3, "n":1, "t":9}
sorted_x = sorted(x.items(), key=operator.itemgetter(1))
print sorted_x

