import re
from gensim.models import Word2Vec
import csv
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



MyMiRBase = []
read_csv(MyMiRBase, 'circBaseSequence.csv')

miRNACorpus = []
counter = 0
while counter < len(MyMiRBase):
    row = re.findall(r'.{0}', MyMiRBase[counter][1])
    miRNACorpus.append(row)
    counter = counter + 1

model = Word2Vec(miRNACorpus, min_count=1, size=64)

miRNAEmbedding = []
counter = 0
while counter < len(list(model.wv.vocab)):
    row = []
    row.append(list(model.wv.vocab)[counter])
    row.extend(model[list(model.wv.vocab)[counter]])
    miRNAEmbedding.append(row)
    counter = counter + 1

store_csv(miRNAEmbedding, 'circRNAEmbedding.csv')

model.save('circRNAModel')
