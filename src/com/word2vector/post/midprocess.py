#coding:gb2312
'''
Created on 2015年1月28日
@author: Administrator
'''
from com.dir import raw
if __name__ == "__main__":
    testraw = open(raw.test_raw_file, "r")
#    tagfile = open(raw.J_biofile, "r")
#    tagfile = open(raw.E_biofile, "r")
#    tagfile = open(raw.biofile, "r")
    tagfile = open(raw.My_biofile, "r")
    predict = open(raw.result, "w")
    for eachline in tagfile:
        oneline = eachline.strip()
        tags = oneline.split()
        print "tags length = ", len(tags)
        pos = 0   
        for testline in testraw:
            testToken = testline.strip()
            if testToken == "":break
            predict.write(testToken + "\t" + tags[pos] + "\n")
            pos += 1
        predict.write("\n")
    predict.close()
    tagfile.close()
    testraw.close()