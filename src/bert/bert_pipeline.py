import argparse
import csv
import os
import torch
from transformers import BertTokenizer

from src.bert.dataloader import GenericDataLoader, QNLData, CoLAData, SST2Data, CoLADataReligion, QNLDataReligion
from src.bert.models import EmbeddingModel

BATCHSIZE = 8

def gender_run():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=False)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_path = args.data_path
    out_path = get_output_path(args)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = CoLAData(tokenizer=tokenizer, data_path=args.data_path)
    data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=8)
    model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE, device=device)

    mean_differences = torch.zeros(768,device=device)
      try:
        for n, el in enumerate(data_loader.train_loader):
            original_sentences = el["original"]
            male_embeddings = model.forward(**el["male"])[1].detach()
            female_embeddings = model.forward(**el["female"])[1].detach()
            # mean_differences = mean_differences*n/(n+1)+(male_embeddings-female_embeddings).mean(0)/(n+1)

    except IndexError:
        pass
        # print(mean_differences)
        # print(n)

def religion_run():
    # parse arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = QNLDataReligion(tokenizer=tokenizer,
                       data_path="D:/Dokumente/Universitaet/Statistik/ML/NLP_new/debiasing-sent/data"
                                 "/QNLI/dev.tsv")
    data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=8)
    model = EmbeddingModel("bert-base-uncased", batch_size=8)
    for el in data_loader.train_loader:
        christ_embedding = model.forward(**el["christ"])
        jew_embedding = model.forward(**el["jew"])
        muslim_embedding = model.forward(**el["muslim"])




if __name__ == "__main__":
    gender_run()
