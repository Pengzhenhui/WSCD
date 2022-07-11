from numpy import *
import numpy as np
import random
import math
import os
import time
import pandas as pd
import csv
import math
import random
import copy


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


if __name__ == '__main__':

    AllDrug = []
    ReadMyCsv(AllDrug, 'RNAInterAllCircRNA.csv')
    AllMi = []
    ReadMyCsv(AllMi, 'RNAInterAllMiRNA.csv')

    AllNode = []
    AllNode.extend(AllDrug)
    AllNode.extend(AllMi)

    print(len(AllNode))


    NodeEmbeddingName = 'SDNEBehavior.txt'
    NodeEmbedding = np.loadtxt(NodeEmbeddingName, dtype=str, skiprows=1)


    # Behavior
    AllNodeBehavior = []
    counter = 0
    while counter < len(AllNode):
        pair = []
        pair.append(AllNode[counter][0])
        counter1 = 0
        while counter1 < len(NodeEmbedding[0]) - 1:
            pair.append(0)
            counter1 = counter1 + 1
        AllNodeBehavior.append(pair)
        counter = counter + 1

    print(np.array(AllNodeBehavior).shape)

    counter = 0
    while counter < len(NodeEmbedding):
        counter1 = 0
        while counter1 < len(AllNode):
            if AllNode[counter1][0] == NodeEmbedding[counter][0]:
                break
            counter1 = counter1 + 1

        # print('counter', counter)
        # print('counter1', counter1)

        AllNodeBehavior[counter1][1:] = NodeEmbedding[counter][1:]
        counter = counter + 1

    print(np.array(AllNodeBehavior).shape)

    #
    StorFile(AllNodeBehavior, 'AllNodeBehaviorSDNE.csv')

