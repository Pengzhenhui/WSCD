import matplotlib.pyplot as plt
import numpy as np
from PIL.EpsImagePlugin import split
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

def ReadMyCsv1(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):
            row[i] = float(row[i])
        SaveList.append(row)
    return

def ReadMyCsv3(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 1
        while counter < len(row):
            row[counter] = float(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return
def MyLabel(Sample):
    label = []
    for i in range(int(len(Sample) / 2)):
        label.append(1)
    for i in range(int(len(Sample) / 2)):
        label.append(0)
    return label


def GenerateFeature(pairs,encode1,encode2):
    sampel_feature = []
    print("nihao",len(pairs))
    for i in range(len(pairs)):
        drug = pairs[i][0]
        mirna = pairs[i][1]
        print(len(encode1))
        print(len(encode2))
        temp = []
        for a in range(len(AllDrugCanonicalSMILES)):
            if drug == AllDrugCanonicalSMILES[a][0]:
                temp.extend(encode1[a][1:])
                print(temp)
        for b in range(len(AllMiSequence)):
            if mirna == AllMiSequence[b][0]:
                temp.extend(encode2[b][1:])
        sampel_feature.append(temp)
    return sampel_feature
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def GenerateEmbeddingFeature(SequenceList, EmbeddingList, PaddingLength):
    SampleFeature = []

    counter = 0
    while counter < len(SequenceList):
        PairFeature = []
        PairFeature.append(SequenceList[counter][0])

        FeatureMatrix = []
        counter1 = 0
        while counter1 < PaddingLength:
            row = []
            counter2 = 0
            while counter2 < len(EmbeddingList[0]) - 1:
                row.append(0)
                counter2 = counter2 + 1
            FeatureMatrix.append(row)
            counter1 = counter1 + 1

        try:
            counter3 = 0
            while counter3 < PaddingLength:
                counter4 = 0
                while counter4 < len(EmbeddingList):
                    if SequenceList[counter][1][counter3] == EmbeddingList[counter4][0]:
                        FeatureMatrix[counter3] = EmbeddingList[counter4][1:]
                        break
                    counter4 = counter4 + 1
                counter3 = counter3 + 1
        except:
            pass

        result_info = np.array(FeatureMatrix)
        irisPca = PCA(n_components=1)
        pcaDate = irisPca.fit_transform(result_info)
        Date = pcaDate.ravel().reshape(1,64)


        PairFeature.append(Date[0].tolist())

        SampleFeature.append(PairFeature)
        counter = counter + 1
    return SampleFeature

def GenerateSampleFeature(InteractionList, EmbeddingFeature1, EmbeddingFeature2):
    SampleFeature1 = []
    SampleFeature2 = []

    counter = 0
    while counter < len(InteractionList):
        Pair1 = InteractionList[counter][0]
        Pair2 = InteractionList[counter][1]

        counter1 = 0
        while counter1 < len(EmbeddingFeature1):
            if EmbeddingFeature1[counter1][0] == Pair1:
                SampleFeature1.append(EmbeddingFeature1[counter1][1])
                break
            counter1 = counter1 + 1

        counter2 = 0
        while counter2 < len(EmbeddingFeature2):
            if EmbeddingFeature2[counter2][0] == Pair2:
                SampleFeature2.append(EmbeddingFeature2[counter2][1])
                break
            counter2 = counter2 + 1

        counter = counter + 1
    SampleFeature1, SampleFeature2 = np.array(SampleFeature1), np.array(SampleFeature2)
    SampleFeature1.astype('float32')
    SampleFeature2.astype('float32')
    print('1SampleFeature1',np.array(SampleFeature1).shape)
    print('2SampleFeature2',np.array(SampleFeature2).shape)


    return SampleFeature1, SampleFeature2

def GenerateBehaviorFeature(InteractionPair, NodeBehavior):
    SampleFeature1 = []
    SampleFeature2 = []
    for i in range(len(InteractionPair)):
        Pair1 = InteractionPair[i][0]
        Pair2 = InteractionPair[i][1]

        for m in range(len(NodeBehavior)):
            if Pair1 == NodeBehavior[m][0]:
                SampleFeature1.append(NodeBehavior[m][1:])
                break

        for n in range(len(NodeBehavior)):
            if Pair2 == NodeBehavior[n][0]:
                SampleFeature2.append(NodeBehavior[n][1:])
                break

    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')
    return SampleFeature1, SampleFeature2

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])


    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


    if test_idx:

        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='y',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')

def TestOutput(classifier, name):

    ModelTestOutput = classifier.predict_proba(X_test)
    ModelTestOutput1 = classifier.predict(X_test)
    LabelPredictionProb = []
    LabelPrediction = []

    counter = 0
    CounterT = 1
    while counter < len(np.array(y_test_Pre)):
        rowProb = []
        rowProb.append(y_test_Pre[counter])
        if y_test_Pre[counter] > 0.5:
            rowProb.append(ModelTestOutput[counter][1])
        else:
            rowProb.append(ModelTestOutput[counter][0])
        LabelPredictionProb.append(rowProb)

        row = []
        row.append(y_test_Pre[counter])
        row.append(ModelTestOutput1[counter])
        LabelPrediction.append(row)

        counter = counter + 1
    LabelPredictionProbName = str(name) + 'RealAndPredictionProbA+B' + str(CounterT) + '.csv'
    StorFile(LabelPredictionProb, LabelPredictionProbName)
    LabelPredictionName = str(name) + 'RealAndPredictionA+B' + str(CounterT) + '.csv'
    StorFile(LabelPrediction, LabelPredictionName)

if __name__ == '__main__':

    AllMiSequence = []
    ReadMyCsv1(AllMiSequence, '../RNAInterAllMiSequence.csv')
    miRNAEmbedding = []
    ReadMyCsv3(miRNAEmbedding, '../miRNAEmbedding.csv')

    AllDrugCanonicalSMILES = []
    ReadMyCsv1(AllDrugCanonicalSMILES, '../RNAInterAllCircBankSMILES.csv')
    DrugEmbedding = []
    ReadMyCsv3(DrugEmbedding, '../circRNAEmbedding.csv')





    AllNodeBehavior = []
    ReadMyCsv1(AllNodeBehavior,
               '../AllNodeBehaviorSDNE.csv')

    PositiveSample_Train = []
    ReadMyCsv1(PositiveSample_Train, 'Positive_Sample_Train1.csv')
    PositiveSample_Validation = []
    ReadMyCsv1(PositiveSample_Validation, 'Positive_Sample_Validation1.csv')
    PositiveSample_Test = []
    ReadMyCsv1(PositiveSample_Test, 'Positive_Sample_Test1.csv')

    NegativeSample_Train = []
    ReadMyCsv1(NegativeSample_Train, 'Negative_Sample_Train1.csv')
    NegativeSample_Validation = []
    ReadMyCsv1(NegativeSample_Validation, 'Negative_Sample_Validation1.csv')
    NegativeSample_Test = []
    ReadMyCsv1(NegativeSample_Test, 'Negative_Sample_Test1.csv')

    x_train_pair = []
    x_train_pair.extend(PositiveSample_Train)
    x_train_pair.extend(NegativeSample_Train)

    x_validation_pair = []
    x_validation_pair.extend(PositiveSample_Validation)
    x_validation_pair.extend(NegativeSample_Validation)

    x_test_pair = []
    x_test_pair.extend(PositiveSample_Test)
    x_test_pair.extend(NegativeSample_Test)


    y_train_Pre = MyLabel(x_train_pair)
    y_validation_Pre = MyLabel(x_validation_pair)
    y_test_Pre = MyLabel(x_test_pair)


    DrugEmbeddingFeature = GenerateEmbeddingFeature(AllDrugCanonicalSMILES, DrugEmbedding, 64)
    miRNAEmbeddingFeature = GenerateEmbeddingFeature(AllMiSequence, miRNAEmbedding, 64)


    x_train_1_Attribute, x_train_2_Attribute = GenerateSampleFeature(x_train_pair, DrugEmbeddingFeature,miRNAEmbeddingFeature)
    x_validation_1_Attribute, x_validation_2_Attribute = GenerateSampleFeature(x_validation_pair, DrugEmbeddingFeature,miRNAEmbeddingFeature)
    x_test_1_Attribute, x_test_2_Attribute = GenerateSampleFeature(x_test_pair, DrugEmbeddingFeature,miRNAEmbeddingFeature)

    x_train_1_Behavior, x_train_2_Behavior = GenerateBehaviorFeature(x_train_pair, AllNodeBehavior)
    x_validation_1_Behavior, x_validation_2_Behavior = GenerateBehaviorFeature(x_validation_pair, AllNodeBehavior)
    x_test_1_Behavior, x_test_2_Behavior = GenerateBehaviorFeature(x_test_pair, AllNodeBehavior)
    print('x_train_1_Attribute','x_train_2_Attribute','x_train_1_Behavior','x_train_2_Behavior',np.array(x_train_1_Attribute).shape,np.array(x_train_2_Attribute).shape,x_train_1_Behavior.shape,x_train_2_Behavior.shape)


    print(x_train_1_Attribute.shape)
    print(x_train_1_Behavior.shape)
    print(x_train_2_Attribute.shape)
    print(x_train_2_Behavior.dtype)
    X_train = []
    X_test = []
    X_validation = []
    for d in range(len(x_train_pair)):
        X_train = np.concatenate((x_train_1_Attribute, x_train_2_Attribute, x_train_1_Behavior, x_train_2_Behavior), axis=1)
    for d in range(len(x_train_pair)):
        X_test = np.concatenate((x_test_1_Attribute, x_test_2_Attribute, x_test_1_Behavior, x_test_2_Behavior), axis=1)

    forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.5, n_estimators=75, n_jobs=4,
            oob_score=False, random_state=None, verbose=1,
            warm_start=False)
    forest.fit(X_train,np.array(y_train_Pre))




    TestOutput(forest, 'RF_')



    GBDT_clf = GradientBoostingClassifier(n_estimators=5)
    GBDT_clf.fit(X_train, np.array(y_train_Pre))
    TestOutput(GBDT_clf, 'GBDT_')


    LR_clf = LogisticRegression()
    LR_clf.fit(X_train, np.array(y_train_Pre))

    TestOutput(LR_clf, 'LR_')

    SVM_clf = svm.SVC(probability = True)
    SVM_clf.fit(X_train,np.array(y_train_Pre))
    TestOutput(SVM_clf, 'SVM_')

    KNN_clf = KNeighborsClassifier()
    KNN_clf.fit(X_train,np.array(y_train_Pre))
    TestOutput(KNN_clf, 'KNN_')

