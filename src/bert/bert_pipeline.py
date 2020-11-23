import argparse
from transformers import BertTokenizer
from src.bert.dataloader import GenericDataLoader, QNLData, CoLAData, SST2Data, CoLADataReligion, QNLDataReligion
from src.bert.models import EmbeddingModel
import torch
import numpy as np
from src.bert.evaluation import female_male_saving


def gender_run():
    # parse arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    batch_size = 8
    data_path = "D:/Dokumente/Universitaet/Statistik/ML/NLP_new/debiasing-sent/data/"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for data in ["CoLA", "QNLI", "SST2"]:
        if data == "CoLA":
            dataset = CoLAData(tokenizer=tokenizer,
                               data_path=data_path + "CoLA/train.tsv")
        elif data == "QNLI":
            dataset = QNLData(tokenizer=tokenizer,
                              data_path=data_path + "QNLI/train.tsv")
        elif data == "SST2":
            dataset = SST2Data(tokenizer=tokenizer,
                               data_path=data_path + "SST-2/train.tsv")
        data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=batch_size)
        model = EmbeddingModel("bert-base-uncased", batch_size=8, device=device)
        mean_difference = torch.zeros(768, device=device)
        result_array_male = np.empty((len(data_loader.train_loader), 768), dtype=float)
        result_array_female = np.empty((len(data_loader.train_loader), 768), dtype=float)

        try:
            for n, el in enumerate(data_loader.train_loader):
                #TODO: add normalization 
                male_embedding = model.forward(**el["male"])[1].detach()
                female_embedding = model.forward(**el["female"])[1].detach()
                result_array_male[n:n + batch_size] = male_embedding.cpu().numpy()
                result_array_female[n:n + batch_size] = female_embedding.cpu().numpy()
                mean_difference = mean_difference * n / (n + 1) + (male_embedding - female_embedding).mean(0) / (n + 1)
        except IndexError:
            print(mean_difference)
            print(n)
        female_male_saving(result_array_male, result_array_female,data_path,data)


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
