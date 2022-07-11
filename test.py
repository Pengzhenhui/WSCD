import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Lambda, GaussianNoise, BatchNormalization, Reshape, dot, Activation, concatenate, AveragePooling1D, GlobalAveragePooling1D
from keras.engine.topology import Layer
from keras.utils import plot_model
from keras.datasets import mnist
from keras import backend as K
from random import shuffle
from keras.callbacks import ReduceLROnPlateau
import csv
csv.field_size_limit(500 * 1024 * 1024)

import numpy as np
import math


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

        PairFeature.append(FeatureMatrix)
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

    SampleFeature1 = SampleFeature1.reshape(SampleFeature1.shape[0], SampleFeature1.shape[1], SampleFeature1.shape[2], 1)
    SampleFeature2 = SampleFeature2.reshape(SampleFeature2.shape[0], SampleFeature2.shape[1], SampleFeature2.shape[2], 1)
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

def MyLabel(Sample):
    label = []
    for i in range(int(len(Sample) / 2)):
        label.append(1)
    for i in range(int(len(Sample) / 2)):
        label.append(0)
    return label

def scheduler(epoch):

    if epoch % 2 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.6)
        print("lr changed to {}".format(lr * 0.6))
    return K.get_value(model.optimizer.lr)

def MyChange(list):
    List = []
    for i in range(len(list)):
        row = []
        row.append(i)
        row.append(float(list[i]))
        List.append(row)
    return List

if __name__ == '__main__':

    AllMiSequence = []
    ReadMyCsv1(AllMiSequence, '../RNAInterAllMiSequence.csv')
    miRNAEmbedding = []
    ReadMyCsv3(miRNAEmbedding, '../miRNAEmbedding.csv')

    AllDrugCanonicalSMILES = []
    ReadMyCsv1(AllDrugCanonicalSMILES, '../RNAInterAllCircBankSMILES.csv')
    DrugEmbedding = []
    ReadMyCsv3(DrugEmbedding, '../circRNAEmbedding.csv')




    PositiveSample_Train = []
    ReadMyCsv1(PositiveSample_Train, 'Positive_Sample_Train0.csv')
    PositiveSample_Validation = []
    ReadMyCsv1(PositiveSample_Validation, 'Positive_Sample_Validation0.csv')
    PositiveSample_Test = []
    ReadMyCsv1(PositiveSample_Test, 'Positive_Sample_Test0.csv')

    NegativeSample_Train = []
    ReadMyCsv1(NegativeSample_Train, 'Negative_Sample_Train0.csv')
    NegativeSample_Validation = []
    ReadMyCsv1(NegativeSample_Validation, 'Negative_Sample_Validation0.csv')
    NegativeSample_Test = []
    ReadMyCsv1(NegativeSample_Test, 'Negative_Sample_Test0.csv')

    x_train_pair = []
    x_train_pair.extend(PositiveSample_Train)
    x_train_pair.extend(NegativeSample_Train)

    x_validation_pair = []
    x_validation_pair.extend(PositiveSample_Validation)
    x_validation_pair.extend(NegativeSample_Validation)

    x_test_pair = []
    x_test_pair.extend(PositiveSample_Test)
    x_test_pair.extend(NegativeSample_Test)

    DrugEmbeddingFeature = GenerateEmbeddingFeature(AllDrugCanonicalSMILES, DrugEmbedding, 64)
    miRNAEmbeddingFeature = GenerateEmbeddingFeature(AllMiSequence, miRNAEmbedding, 25)

    x_train_1_Attribute, x_train_2_Attribute = GenerateSampleFeature(x_train_pair, DrugEmbeddingFeature, miRNAEmbeddingFeature)
    x_validation_1_Attribute, x_validation_2_Attribute = GenerateSampleFeature(x_validation_pair, DrugEmbeddingFeature, miRNAEmbeddingFeature)
    x_test_1_Attribute, x_test_2_Attribute = GenerateSampleFeature(x_test_pair, DrugEmbeddingFeature, miRNAEmbeddingFeature)

    y_train_Pre = MyLabel(x_train_pair)
    y_validation_Pre = MyLabel(x_validation_pair)
    y_test_Pre = MyLabel(x_test_pair)
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train_Pre, num_classes)
    y_validation = keras.utils.to_categorical(y_validation_Pre, num_classes)
    y_test = keras.utils.to_categorical(y_test_Pre, num_classes)


    print('x_train_1_Attribute shape', x_train_1_Attribute.shape)
    print('x_train_2_Attribute shape', x_train_2_Attribute.shape)


    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    CounterT = 0



    input1 = Input(shape=(len(x_train_1_Attribute[0]), len(x_train_1_Attribute[0][0]), 1), name='input1')
    x1 = Conv2D(64, kernel_size=(4, 64), activation='relu', name='conv1')(input1)
    x1 = MaxPooling2D(pool_size=(1, 1), name='pool1')(x1)
    x1 = Flatten()(x1)

    x1 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.0001))(x1)
    x1 = Dropout(rate=0.3)(x1)

    input2 = Input(shape=(len(x_train_2_Attribute[0]), len(x_train_2_Attribute[0][0]), 1), name='input2')
    x2 = Conv2D(64, kernel_size=(4, 64), activation='relu', name='conv2')(input2)
    x2 = MaxPooling2D(pool_size=(1, 1), name='pool2')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.0001))(x2)
    x2 = Dropout(rate=0.3)(x2)

    flatten = keras.layers.concatenate([x1, x2])

    hidden = Dense(32, activation='relu', name='hidden2', activity_regularizer=regularizers.l2(0.0001))(flatten)
    hidden = Dropout(rate=0.3)(hidden)
    output = Dense(num_classes, activation='softmax', name='output')(hidden)
    model = Model(inputs=[input1, input2], outputs=output)


    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])



    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,patience=1, mode='min', min_delta=0.000000001)
    history = model.fit({'input1': x_train_1_Attribute, 'input2': x_train_2_Attribute}, y_train,
                        validation_data=({'input1': x_validation_1_Attribute, 'input2': x_validation_2_Attribute,
                                          }, y_validation),
                        callbacks=[reduce_lr],
                        epochs=50, batch_size=128,
                        )


    ModelName = 'my_model' + str(CounterT) + '.h5'
    model.save(ModelName)


    ModelTest = Model(inputs=model.input, outputs=model.get_layer('output').output)
    ModelTestOutput = ModelTest.predict(
        [x_test_1_Attribute, x_test_2_Attribute])
    print(ModelTestOutput.shape)
    print(type(ModelTestOutput))
    LabelPredictionProb = []
    LabelPrediction = []

    counter = 0
    while counter < len(ModelTestOutput):
        rowProb = []
        rowProb.append(y_test_Pre[counter])
        rowProb.append(ModelTestOutput[counter][1])
        LabelPredictionProb.append(rowProb)

        row = []
        row.append(y_test_Pre[counter])
        if ModelTestOutput[counter][1] > 0.5:
            row.append(1)
        else:
            row.append(0)
        LabelPrediction.append(row)

        counter = counter + 1
    LabelPredictionProbName = 'RealAndPredictionProbA+B' + str(CounterT) + '.csv'
    StorFile(LabelPredictionProb, LabelPredictionProbName)
    LabelPredictionName = 'RealAndPredictionA+B' + str(CounterT) + '.csv'
    StorFile(LabelPrediction, LabelPredictionName)

