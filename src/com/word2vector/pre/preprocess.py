#coding:gb2312
'''
Created on 2015��1��26��
@author: Administrator
'''
from com.word2vector.dir import raw
'''
    ������ļ������Ѿ���ȡ��������trian��test��������ļ���ȥ���ļ��е�������
'''
def tokenFile(filename):
    if filename.find("train") != -1:
        file = open(filename, "r")
        rawfile = open(raw.train_raw_file, "w")
        for eachline in file:
            oneline = eachline.strip()
            if oneline == "":
                rawfile.write("\n")
                continue
            lineToken = oneline.split()
            rawfile.write(lineToken[0]+"\t"+lineToken[-1]+"\n")
        file.close()
        rawfile.close()
    elif filename.find("test") != -1:
        file = open(filename, "r")
        rawfile = open(raw.test_raw_file, "w")
        for eachline in file:
            oneline = eachline.strip()
            if oneline == "":
                rawfile.write("\n")
                continue
            lineToken = oneline.split()
            rawfile.write(lineToken[0]+"\n")
        file.close()
        rawfile.close()
    else:
        assert "pls edit your raw filename again !"

if __name__ == "__main__":
    print "start ..."
    tokenFile(raw.train_file)
    tokenFile(raw.test_file)