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
        if(min_ == max_):
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


def train(epoch, index, cutting, adjacentMatrix, feature, label, trainIndex):
    lossSet = []
    batch = int(len(trainIndex)/cutting)
    for i in range(epoch):
        for j in range(cutting):
            if torch.cuda.is_available():
                indexPerBatch = trainIndex[index[j*batch:(j+1)*batch]].cuda()
            else:
                indexPerBatch = trainIndex[index[j*batch:(j+1)*batch]]
            optimizer.zero_grad()
            output = model(feature, adjacentMatrix)
            loss = criterion(output[indexPerBatch], label[indexPerBatch])
            loss.backward()
            optimizer.step()
        lossSet.append(loss.item())
    return lossSet


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
    _parser.add_argument("--cutting",
                         default = 3,
                         help = "Batch = dataNum/cutting", 
                        type=int)
    _parser.add_argument("--epoch",
                         default = 1000,
                         help = "Epoch", 
                        type=int)
    _parser.add_argument("--numHeads",
                         default = 4,
                         help = "numHeads", 
                        type=int)
    _args = _parser.parse_args()
    
    _withSuspectValue, _run, _cutting, _epoch, _numHeads = _args.withSuspectValue, _args.run, _args.cutting, _args.epoch, _args.numHeads
    _adjacentMatrix, _featureSet, _label = loadData(_withSuspectValue)
    _trainIndex, _testIndex, _updateLossIndex = loadIndex(35681, 44602)
    
    for _i in range(_run):   
        _startTime = time.time()
        _savePath = './GAT_' + str(_i) + '.pth'
        model = net(_numHeads, _withSuspectValue)
        if torch.cuda.is_available():
            model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        _lossSet = train(_epoch, _updateLossIndex, _cutting, _adjacentMatrix, _featureSet, _label, _trainIndex)
        
        torch.save(model, _savePath)
        _endTime = time.time()
        print(_i, _savePath, "done")
