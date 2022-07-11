import numpy as np
from sklearn.model_selection import KFold
import csv
import random
import pandas as pd

def read_csv(save_list, file_name):
    csv_reader = csv.reader(open(file_name))
    for row in csv_reader:
        save_list.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return
def generate_negative_sample(relationship_pd):
    relationship_matrix = pd.pivot_table(relationship_pd, index='Pair1', columns='Pair2', values='Rating', fill_value=0)
    negative_sample = []
    counter = 0
    while counter < len(relationship_pd):

        temp_1 = random.randint(0, len(relationship_matrix.index) - 1)
        temp_2 = random.randint(0, len(relationship_matrix.columns) - 1)
        if relationship_matrix.iloc[temp_1, temp_2] == 0:
            relationship_matrix.iloc[temp_1, temp_2] = -1
            row = []
            row.append(np.array(relationship_matrix.index).tolist()[temp_1])
            row.append(np.array(relationship_matrix.columns).tolist()[temp_2])
            negative_sample.append(row)

            counter = counter + 1

        else:
            pass

    return negative_sample, relationship_matrix

if __name__ == '__main__':
    relationship = []
    read_csv(relationship, 'RNAInterCircMiInteraction.csv')
    random.shuffle(relationship)



    Sam = np.array(relationship)
    New_sam = KFold(n_splits=5)
    CounterT = 0
    for train_index, test_index in New_sam.split(Sam):
        Sam_train, Sam_test = Sam[train_index], Sam[test_index]
        TrainName = 'Positive_Sample_Train' + str(CounterT) + '.csv'
        StorFile(Sam_train, TrainName)
        New_sam_test = KFold(n_splits=2)
        for validation_index, test_index_1 in New_sam_test.split(Sam_test):
            Sam_validation, Sam_TTest = Sam_test[validation_index], Sam_test[test_index_1]
            ValidationName = 'Positive_Sample_Validation' + str(CounterT) + '.csv'
            StorFile(Sam_validation, ValidationName)
            TestName = 'Positive_Sample_Test' + str(CounterT) + '.csv'
            StorFile(Sam_TTest, TestName)
        CounterT = CounterT + 1


    relationship_pd = pd.DataFrame(relationship, columns=['Pair1', 'Pair2'])
    relationship_pd['Rating'] = [1] * len(relationship_pd)  # ？？？？？？如何理解？ 向下



    negative_sample, relationship_matrix = generate_negative_sample(relationship_pd)
    relationship_matrix.to_csv('Relationship_Matrix_demo.csv')
    print(negative_sample)
    Sam2 = np.array(negative_sample)
    print(Sam2.shape)

    New_sam2 = KFold(n_splits=5)
    CounterT2 = 0
    for train_index2, test_index2 in New_sam2.split(Sam2):
        Sam_train2, Sam_test2 = Sam2[train_index2], Sam2[test_index2]
        TrainName2 = 'Negative_Sample_Train' + str(CounterT2) + '.csv'
        StorFile(Sam_train2, TrainName2)
        New_sam_test2 = KFold(n_splits=2)
        for validation_index2, test_index_12 in New_sam_test2.split(Sam_test2):
            Sam_validation2, Sam_TTest2 = Sam_test2[validation_index2], Sam_test2[test_index_12]
            ValidationName2 = 'Negative_Sample_Validation' + str(CounterT2) + '.csv'
            StorFile(Sam_validation2, ValidationName2)
            TestName2 = 'Negative_Sample_Test' + str(CounterT2) + '.csv'
            StorFile(Sam_TTest2, TestName2)
        CounterT2 = CounterT2 + 1