import argparse
import os
import csv


def read_tsv(file_name,delimiter="\t"):
    tsv_file = open(file_name, encoding="utf8")
    tsv_reader = csv.reader(tsv_file, delimiter=delimiter)
    # tsv_reader.__next__()
    lines = []
    for line in tsv_reader:
        lines.append(line)
    return lines

def preprocess(data_path):
    file_name = os.path.join(data_path,"datasetSentences.txt")
    sentence_list = read_tsv(file_name)
    file_name = os.path.join(data_path,"datasetSplit.txt")
    label_list = read_tsv(file_name)
    file_name = os.path.join(data_path,"sentiment_labels.txt")
    label_list = read_tsv(file_name,"|")
    print(sentence_list[0])
    print(label_list[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default="./data/", type=str, required=False)
    args = parser.parse_args()

    data_path = os.path.join(args.data_path, "SST-2-raw")
    preprocess(data_path)


