import argparse
import os
import json
import time
import random
import torch
import dgl
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from dgl.nn import GATConv
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score

class net(nn.Module):
    def __init__(self, num_heads, withSuspectValue):
        super(net, self).__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.num_heads = num_heads
        
        if _withSuspectValue:
            self.GATC1 = GATConv(73, 32, num_heads=num_heads)
        else:
            self.GATC1 = GATConv(72, 32, num_heads=num_heads)
        self.fc1 = nn.Linear(32*num_heads, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 2)
        self.batchNorm_1 = nn.BatchNorm1d(32*num_heads, affine=True)
        self.batchNorm_2 = nn.BatchNorm1d(32, affine=True)
        self.batchNorm_3 = nn.BatchNorm1d(8, affine=True)
        self.batchNorm_4 = nn.BatchNorm1d(4, affine=True)
        self.batchNorm_5 = nn.BatchNorm1d(2, affine=True)
        self.Softmax = nn.Softmax(dim=1)
        
    def forward(self, x, adj):
        x = self.GATC1(adj, x)
        x = x.view(-1, 32*self.num_heads)
        x = self.relu(self.batchNorm_1(x))
        x = self.fc1(x)
        x = self.relu(self.batchNorm_2(x))
        x = self.fc2(x)
        x = self.relu(self.batchNorm_3(x))
        x = self.fc3(x)
        x = self.relu(self.batchNorm_4(x))
        x = self.fc4(x)
        x = self.relu(self.batchNorm_5(x))
        x = self.Softmax(x)
        return x
    
def normalization(array):
    for i in range(len(array[0])):
        min_ = array[0][i].min()
        max_ = array[0][i].max()
        if(min_==max_):
            continue
        for j in range(len(array[0][0])):
            array[0][i][j] = np.around((array[0][i][j]-min_)/(max_-min_), decimals=5)
    return array


def loadJson(path, file):
    f = open(path+file+".json", 'r')
    for line in f:
        jsonFile = json.loads(line)
    f.close()
    return jsonFile


def loadIndex(trainBoundary, testBoundary):
    trainIndex = torch.LongTensor(range(trainBoundary))
    testIndex = torch.LongTensor(range(trainBoundary, testBoundary))
    updateLossIndex = [i for i in range(trainBoundary)]  
    return trainIndex, testIndex, updateLossIndex


def getFeature(withSuspectValue):
    infoPath = './data/information/'
    numpyPath = './data/userSet/'
    userSet = loadJson(infoPath, 'userIndex')
    userLabel = loadJson(infoPath, 'userLabel')
    suspectValue = loadJson(infoPath, 'suspectValue')
    
    featureSet, label = [], []
    for user in userSet:
        suspect = suspectValue[user]
        if userLabel[user]==0:
            label.append(0)
        else:
            label.append(1)
        user = normalization(np.load(numpyPath+user+r'.npy')).reshape((72))
        if withSuspectValue:
            user = np.append(user, suspect)
        featureSet.append(user)
    return featureSet, label


def loadData(withSuspectValue):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = "./data/information/"
    adjacentMatrix = sp.load_npz(path+"adjacentMatrix.npz")
    adjacentMatrix = dgl.from_scipy(adjacentMatrix).to(device)
    featureSet, label = getFeature(withSuspectValue)
    featureSet = torch.from_numpy(np.array(featureSet)).float().to(device)
    label = torch.Tensor(label).long().to(device)
    return adjacentMatrix, featureSet, label


def test(adjacentMatrix, feature):
    with torch.no_grad():
        output = model(feature, adjacentMatrix)
    return output


def getAUC(state, output, lower, upper):
    path = "./data/information/"
    activeValue = loadJson(path, 'activeValue')
    indexUser = loadJson(path, 'indexUser')
    spammer = loadJson(path, 'spammer')
    userLabel = loadJson(path, 'userLabel')
    confidenceSet, labelSet = [], []
    for i in range(35681, len(output)):
        if lower<=activeValue[indexUser[str(i)]] and activeValue[indexUser[str(i)]]<=upper:
            confidenceSet.append(float(output[i][1])) 
            labelSet.append(userLabel[indexUser[str(i)]])
    if state == "ROC":
        fpr, tpr, threshold = roc_curve(labelSet, confidenceSet)
        aucValue = auc(fpr, tpr)
    elif state == "PR":
        precision, recall, thresholds = precision_recall_curve(labelSet, confidenceSet)
        aucValue = auc(recall, precision)
    return aucValue


def getF1score(output, lower, upper, k):
    dic, candidate = {}, []
    path = './data/information/'
    indexUser = loadJson(path, 'indexUser')
    userIndex = loadJson(path, 'userIndex')
    userLabel = loadJson(path, 'userLabel')
    activeValue = loadJson(path, 'activeValue')
    spammer = loadJson(path, 'spammer')
    output = np.array(output.cpu())
    for i in range(35681, len(output)):
        user = indexUser[str(i)]
        dic[user] = output[i][1]
    sort = sorted(dic.items(), key=lambda x:x[1])
    sort = sort[(8921-k):]
    for cell in sort:
        if lower<=activeValue[cell[0]] and activeValue[cell[0]]<upper:
            candidate.append(cell[0])
    tp, fp = 0, 0
    for user in candidate:
        if userLabel[user]==1:
            tp+=1
        elif userLabel[user]==0:
            fp+=1
    
    try:
        precision, recall = tp/(tp+fp), tp/183 
    except:
        precision, recall = 0, tp/183 
    try:
        f1score = (2*precision*recall)/(precision+recall)
    except:
        f1score = 0
    return precision, recall, f1score


if __name__ == "__main__":   
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--withSuspectValue", 
                         default = True,
                         help = "Use Suspect Value", 
                        type=bool)
    _parser.add_argument("--run",
                         default = 10,
                         help = "Run", 
                        type=int)
    _parser.add_argument("--numHeads",
                         default = 4,
                         help = "numHeads", 
                        type=int)
    _args = _parser.parse_args()
    
    _withSuspectValue, _run, _numHeads = _args.withSuspectValue, _args.run, _args.numHeads
    
    _withSuspectValue = True
    _adjacentMatrix, _featureSet, _label = loadData(_withSuspectValue)
    
    _aurocSetALL, _auprcSetALL = [], []
    _intervals = [[0, 18], [18, 45], [45, 84], [84, 135], [135, 211], [211, 315], [315, 494], [494, 817], [817, 1663], 
                [1663, 100000000000000000000000000]]
    
    print("-------------------AUROC/AUPRC-------------------")
    for _interval in _intervals:
        _lower, _upper = _interval[0], _interval[1]
        _aurocSet, _auprcSet = [], []
        
        for _i in range(_run):
            modelPath = './GAT_' + str(_i) + '.pth'
            model = torch.load(modelPath)  
            _output = test(_adjacentMatrix, _featureSet)
            _auroc = getAUC("ROC", _output, _lower, _upper)
            _auprc = getAUC("PR", _output, _lower, _upper)
            _aurocSet.append(_auroc)
            _auprcSet.append(_auprc)
        
        print(time.asctime(time.localtime(time.time())), _interval, "done")
        print("AUROC :", np.around(np.median(_aurocSet), decimals=4), '±', np.around(np.std(_aurocSet), decimals=4))
        print("AUPRC :", np.around(np.median(_auprcSet), decimals=4), '±', np.around(np.std(_auprcSet), decimals=4))
    
    
    print("-------------------TopK F1score/Recall/Precision-------------------")
    ## TopK F1score/Recall/Precision
    _intervals = [[0, 18], [18, 45], [0, 100000000000000000000000000]]
    _topKset = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    for _i in range(len(_intervals)):
        _lower, _upper = _intervals[_i][0], _intervals[_i][1]
        
        for _topK in _topKset:
            _precisionSet, _recallSet, _f1scoreSet = [], [], []
            
            for _j in range(_run):
                modelPath = './GAT_' + str(_j) + '.pth'
                model = torch.load(modelPath)  
                _output = test(_adjacentMatrix, _featureSet)

                _precision, _recall, _f1score = getF1score(_output, _lower, _upper, _topK)
                _f1scoreSet.append(_f1score)
                _precisionSet.append(_precision)
                _recallSet.append(_recall)
                
            print(time.asctime(time.localtime(time.time())), _intervals[_i], _topK, "done")
            print("F1score   :", np.around(np.median(_f1scoreSet), decimals=4), '±', np.around(np.std(_f1scoreSet), decimals=4))
            print("Recall    :", np.around(np.median(_recallSet), decimals=4), '±', np.around(np.std(_recallSet), decimals=4))
            print("Precision :", np.around(np.median(_precisionSet), decimals=4), '±', np.around(np.std(_precisionSet), decimals=4))