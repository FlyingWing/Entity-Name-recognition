#coding:gb2312
'''
Created on 2015年4月13日
@author: Administrator
'''
def staNum(file1, file2):
    #建立一个存放set的列表
    lst = []
    for i in xrange(14):
        lst.append(set())
    for eachline in file1:
        oneline = eachline.strip()
        for epoch, element in enumerate(oneline.split("\t")[2:-1]):
            lst[epoch].add(element)
    for eachline in file2:
        oneline = eachline.strip()
        for epoch, element in enumerate(oneline.split("\t")[2:]):
            lst[epoch].add(element)
    return [len(iterm) for iterm in lst]
            
        
if __name__ == "__main__":
    a = set()
    str = "Phenotypic Phenotypic JJ __nil__ mor_s:no* v:--e-o---i- lens:6+ mor_o:Aaaaaaaaaa"
    for i, iterm in enumerate(str.split(" ")[0:5]):
        a.add(iterm)
    print len(a)