import numpy as np
import pandas as pd
import csv
import math
import random
import gc
#pandas功能：数据文件读取/文本数据读取；　索引、选取和数据过滤；算法运算和数据对齐；函数的应用和映射；重置索引
csv.field_size_limit(500 * 1024 * 1024)


def read_csv(save_list, file_name):
    csv_reader = csv.reader(open(file_name))
    for row in csv_reader:
        save_list.append(row)
    return


def store_csv(data, file_name):
    with open(file_name, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)
    return


def generate_negative_sample(relationship_pd):
    relationship_matrix = pd.pivot_table(relationship_pd, index='Pair1', columns='Pair2', values='Rating', fill_value=0)
    negative_sample = []
    counter = 0
    while counter < len(relationship_pd):
        #print(counter)
        temp_1 = random.randint(0, len(relationship_matrix.index) - 1) #返回a到b之间的任意数
        temp_2 = random.randint(0, len(relationship_matrix.columns) - 1)
        if relationship_matrix.iloc[temp_1, temp_2] == 0: #利用loc、iloc提取行、列数据
            relationship_matrix.iloc[temp_1, temp_2] = -1
            row = []#list类型
            row.append(np.array(relationship_matrix.index).tolist()[temp_1])#将array类型转换成list类型
            row.append(np.array(relationship_matrix.columns).tolist()[temp_2])
            negative_sample.append(row)

            counter = counter + 1

        else:
            pass

    return negative_sample, relationship_matrix


if __name__ == '__main__':


    #relationship_pd = pd.DataFrame(relationship_matrix111)
    #读取关系矩阵，用pd.read_csv
    relationship_matrix = pd.read_csv('Relationship_Matrix.csv', index_col="Pair1")

    #print(relationship_matrix)
    #print(relationship_matrix.iloc[3,0])


    CS_negative_sample = []
    temp_1 = 0
    # index指的是行3569、colums是列1152
    while temp_1 < len(relationship_matrix.index):
        CS_negative_sample1 = []

        temp_2 = 0
        while temp_2 < len(relationship_matrix.columns):
            #print(relationship_matrix.iloc[temp_1, temp_2])
            if relationship_matrix.iloc[temp_1, temp_2] == 0:  # 利用loc、iloc提取行、列数据
                #print(relationship_matrix.iloc[temp_1, temp_2])
                row = []  # list类型
                row.append(np.array(relationship_matrix.index).tolist()[temp_1])  # 将array类型转换成list类型
                row.append(np.array(relationship_matrix.columns).tolist()[temp_2])
                CS_negative_sample.append(row)
                #CS_negative_sample.extend(CS_negative_sample1)

            temp_2 = temp_2 + 1
        temp_1 = temp_1 + 1
        gc.collect()
    print('CS_negative_sample', len(CS_negative_sample))
    store_csv(CS_negative_sample, 'CaseStudy_negative_sample.csv')
    #CS_negative_sample.to_csv("list.csv")