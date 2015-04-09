#coding:gb2312
'''
Created on 2014年11月4日

@author: Administrator
'''
import os
def processTestCorpus(originalTestFile):
    testset = open(originalTestFile, 'r')
    allLines = testset.readlines()
    testset.close()
    essay = {}
    for eachline in allLines:
        oneline = eachline.strip()
        essay_content = oneline.split(None, 1)
        essay_content_id = essay_content[0]
        essay_content_token = essay_content[1].split()
        essay_content_list = []
        pos_begin = -1
        pos_end   = 0
        for each_token in essay_content_token:
            if isvalid(each_token):
                pos_end = pos_begin + len(each_token)
                pos_begin +=1
                essay_content_list.append([[pos_begin, pos_end], each_token])
                pos_begin = pos_end
            else:
                pos_begin = pos_begin + len(each_token)        
        essay[essay_content_id] = essay_content_list
    return essay

def isvalid(token):
    length = len(token)
    if length > 0 :
        if token[0] == '(' or token[0] == '"':
            if length > 1:
                token = token[1:]
                length = length - 1
            else:
                return False
    else:
        return False    
    while length != 0 and token != None and (token[length-1] == ')' or token[length-1] == ',' or token[length-1] == '.' or token[length-1] == '"' or token[length-1] == ':' or token[length-1] == ';'):
        if length > 1:
            token = token[0:length-1]
            length = length - 1
        else:
            return False
    return True
    
def write_prediction(originalTestCorpus = None, prediction = {}, filename = None):
    testset = open(originalTestCorpus, 'r')
    allLines = testset.readlines()
    testset.close()
    prediction_file = open(filename, "w")
    for eachline in allLines:
        oneline = eachline.strip()
        essay_content = oneline.split(None, 1)
        essay_content_id = essay_content[0]
        essay_content_token = essay_content[1].split()
        essay_content_list = []
        pos_begin = -1
        pos_end   = 0
        for each_token in essay_content_token:
            if isvalid(each_token):
                pos_end = pos_begin + len(each_token)
                pos_begin +=1
                essay_content_list.append([(pos_begin, pos_end), each_token])
                pos_begin = pos_end
            else:
                pos_begin = pos_begin + len(each_token)        
        if essay_content_id in prediction.keys():
            pre_path = prediction[essay_content_id]
            Flag = False
            sentenceID = None
            gene_begin = -1
            gene_end   = 0
            gene_name  = None
            for i in range(len(pre_path)):
                if pre_path[i] == "B":
                    Flag = True
                    sentenceID = essay_content_id
                    current_begin, current_end = essay_content_list[i][0]
                    gene_begin = str(current_begin)
                    gene_end   = str(current_end)
                    gene_name  = essay_content_list[i][1]
                elif pre_path[i] == "O":
                    Flag = False
                    if sentenceID != None:
                        prediction_file.write(sentenceID + "|" + gene_begin + " " + gene_end + "|" + gene_name + "\n")
                        sentenceID = None
                elif Flag :
                    current_begin, current_end = essay_content_list[i][0]
                    gene_end   = str(current_end)
                    gene_name  = gene_name + " " + essay_content_list[i][1]  
            if sentenceID != None:
                prediction_file.write(sentenceID + "|" + gene_begin + " " + gene_end + "|" + gene_name + "\n")
    prediction_file.close()
def post_processing(inputfile = None, outputfile = None):
    readerFile = open(inputfile, "r")
    allLines = readerFile.readlines()
    readerFile.close()
    writerFile = open(outputfile, "w")
    for eachline in allLines:
        oneline = eachline.strip()
        infor_list = oneline.split("|")
        pos_pair  = infor_list[1].split()
        pos_begin = int(pos_pair[0])
        pos_end   = int(pos_pair[1])
        gene_name = infor_list[2]
        i = len(gene_name)-1
        while (gene_name[i] == ',' 
            or gene_name[i] == '.' 
            or gene_name[i] == '"' 
            or gene_name[i] == ':' 
            or gene_name[i] == ';'
            or gene_name[i] == '-'):
            i -= 1
            pos_end -= 1
        length = i + 1
        gene_name = gene_name[0:length]
        if gene_name[0] == "(" and gene_name[length-1] == ")" :
            pos_begin += 1
            pos_end   -= 1
            gene_name = gene_name[1:length-1]
        elif gene_name[0] == "(" and gene_name.find(")") == -1:
            pos_begin += 1
            gene_name = gene_name[1:]
        elif gene_name[length-1] == ")" and gene_name.find("(") == -1 :
            pos_end -=1 
            gene_name =  gene_name[:length-1]
        if gene_name.count("(") == gene_name.count(")"):
            writerFile.write(infor_list[0] + "|" + str(pos_begin) + " " + str(pos_end) + "|" + gene_name + "\n")
    writerFile.close()                                                    
if __name__ == "__main__":
##    processTestCorpus(originalTestCorpus)     
#    list =  ["O", "O", "O", "O", "O", "O", "B", "I", "O", "O", "B", "O", "O", "O", "O", "O", "B"]
#    prediction = {"BC2GM096399526":list}
#    write_prediction(originalTestCorpus, prediction , "test_pre.pkl")
#    currentDir = os.path.dirname(__file__)
    post_processing("test.eval", "test2.eval")
#    str = "alk("
#    print str.count("(")