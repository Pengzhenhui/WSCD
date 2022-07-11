import pandas as pd
from pandas import DataFrame

df = DataFrame([
['a','b','c','d'],
[1,2,3,4]
])

df2 = DataFrame(df,index=['one','two'],columns=['aa','bb','cc','dd'])


print(df2)
print(df2.index)
print(df2.columns)
from pandas import DataFrame
dict1 = dict(aprt=['101', '102', '103'], profits=[1000, 2000, 3000], year=[2001, 2002, 2003], month=8)
df3 = DataFrame(dict1)
df3.index=['one','two','three']
print(df3)
df3.to_csv('data1.csv')
df4=pd.read_csv('data1.csv', sep=';',encoding='UTF-8',header=None)
df4.to_csv('data2.csv')
print(df4)
