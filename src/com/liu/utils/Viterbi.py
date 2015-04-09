# coding:utf8

class MyViterbi():
    def __init__(self, states=['B', 'I', 'O']):
        self.stateMap = {}
        self.indexMap = {}
        index = 0
        for state in states:
            self.stateMap[state] = index
            self.indexMap[index] = state
            index += 1
#         print self.stateMap
        self.matrix = []
        self.pathLog = []
    
    def firstRow(self, probs={'B':0.8, 'I':0, 'O':0.2}):
        probs['I'] = 0
        self.matrix.append([probs[self.indexMap[i]] for i in range(len(self.indexMap))])
        
    def getPath(self):
#         print self.matrix
#         print self.pathLog
        lastRow = self.matrix[len(self.matrix) - 1]
        maxV = lastRow[2]
        lastS = 2
        for i in range(len(lastRow)):
            if lastRow[i] > maxV:
                maxV = lastRow[i]
                lastS = i
#         print lastS
        path = [lastS]
        for p in range(len(self.pathLog) - 1, -1, -1):
            for tup in self.pathLog[p]:
                if tup[1] == lastS:
                    path.append(tup[0])
                    lastS = tup[0]
                    break
        path.reverse()
        return [self.indexMap[i] for i in path]
        
    def addRow(self, probs={('b', 'b'):0.1, 
                            ('b', 'i'):0.5, 
                            ('b', 'o'):0.4, 
                            ('i', 'b'):0.1, 
                            ('i', 'i'):0.4, 
                            ('i', 'o'):0.5, 
                            ('o', 'b'):0.3, 
                            ('o', 'o'):0.7, 
                            ('o', 'i'):0 }):
        probs[('o', 'i')] = 0
        lastRow = self.matrix[len(self.matrix) - 1]
        cRow = []
        cPath = []
        for i in range(len(self.indexMap)):
            maxV = -100
            maxP = None
            for j in range(len(self.indexMap)):
                cv = lastRow[j] * probs[(self.indexMap[j], self.indexMap[i])]
                if maxV < cv:
                    maxV = cv
                    maxP = (j, i)
            cRow.append(maxV)
            cPath.append(maxP)
        self.matrix.append(cRow)
        self.pathLog.append(cPath)
        
if __name__ == '__main__':
    v = MyViterbi()
    v.firstRow()
    v.addRow()
    v.addRow()
    v.addRow()
    v.addRow()
    v.addRow()
    print v.matrix
    print v.pathLog
    print v.getPath()
    
    
        
            
