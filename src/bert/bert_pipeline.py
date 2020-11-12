import argparse
from transformers import BertTokenizer
from src.bert.dataloader import GenericDataLoader, QNLData, CoLAData, SST2Data, CoLADataReligion, QNLDataReligion
from src.bert.models import EmbeddingModel
import torch

def gender_run():
    # parse arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = QNLData(tokenizer=tokenizer,
                       data_path="D:/Dokumente/Universitaet/Statistik/ML/NLP_new/debiasing-sent/data"
                                 "/QNLI/dev.tsv")
    data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=8)
    model = EmbeddingModel("bert-base-uncased", batch_size=8)
    mean_difference = torch.zeros(768)
    for n, el in enumerate(data_loader.train_loader):
        male_embedding = model.forward(**el["male"])[1].detach()
        female_embedding = model.forward(**el["female"])[1].detach()
        mean_difference = mean_difference*n/(n+1)+(male_embedding-female_embedding).mean(0)/(n+1)

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
