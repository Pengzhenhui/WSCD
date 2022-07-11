import numpy as np
import pandas as pd
from itertools import product
from functools import reduce



def Kmers_funct(seq,k):
    X = [None]*len(seq)
    for i in range(len(seq)):
        a = seq[i]
        t=0
        l=[]
        for index in range(len(a)):
            t=a[index:index+k]
            if (len(t))==k:
                l.append(t)
        X[i] = l
    return np.array(X)


def nucleotide_type(k):
    z = []
    for i in product('ACGU', repeat = k):
        z.append(''.join(i))
    return z


def Kmers_frequency(seq,x):
    X = []
    char = reduce(lambda x, y: [i + j for i in x for j in y], [['A', 'T', 'C', 'G']] * x)

    for i in range(len(seq)):
        s = seq[i]
        frequence = []
        for a in char:
            number = s.count(a)

            char_frequence = number/len(s)
            frequence.append(char_frequence)
        X.append(frequence)
    return X

def Kmers_sumfrequency(seq,x,sum2):
    X = []
    sum3 = 0
    sum33 = 0
    char = reduce(lambda x, y: [i + j for i in x for j in y], [['A', 'T', 'C', 'G']] * x)


    for a in char:
        for i in range(len(seq)):
            s = seq[i]
            number = s.count(a)

            sum3 = sum3+number
        print(a,sum3)
        char_frequence = sum3 / sum2
        X.append(char_frequence)
    return X


if __name__ == '__main__':
    k = 6
    miRNA = pd.read_csv(r'circBaseSequence.csv')
    seq = miRNA['SequenceID']
    print(len(seq))
    seq_kmer=Kmers_funct(seq, k)
    print(len(seq_kmer))

    resoult = {}
    sum1 = 0
    sum2 = 0
    j=0
    for i in range(len(seq_kmer)):
        s = seq_kmer[i]
        j = j+1
        for i in s:
            resoult[i] = s.count(i)


        values = resoult.values()
        for value in values:
            sum1 = sum1 + value

        sum2 = sum2+sum1
    print("sum2",sum2)

    feature_kmer=Kmers_frequency(seq_kmer,k)
    Kmers_sumf = Kmers_sumfrequency(seq_kmer,k,sum2)

    pd.DataFrame(feature_kmer).to_csv(r'miRNA_fvector.csv', header=None, index=None)
    pd.DataFrame(seq_kmer).to_csv(r'miRNAseq_Kmers.csv', header=None, index=None)
    pd.DataFrame(Kmers_sumf).to_csv(r'miRNAsum_Kmerf.csv', header=None, index=None)

