import argparse
from transformers import BertTokenizer
from src.bert.dataloader import GenericDataLoader, NewsData, QNLData, CoLAData, SST2Data, CoLADataReligion, \
    QNLDataReligion
from src.bert.models import EmbeddingModel
import torch
import numpy as np
from src.bert.evaluation import female_male_saving, ScoreComputer
from src.bert.dataloader import load_from_database, select_data_set
from src.debiasing.pca import DebiasingPCA
from src.bert.evaluation import male_female_forward_pass, prepare_pca_input

BATCHSIZE = 8


def gender_example_creation(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE, device=device)

    compute_score = ScoreComputer(tokenizer, model, BATCHSIZE, device)

    compute_score.compute_all_metrics()

    for data in ["SST2", "CoLA", "QNLI"]:
        if data == "CoLA":
            dataset = CoLAData(tokenizer=tokenizer,
                               data_path=args.data_path + "CoLA/train.tsv")
        elif data == "QNLI":
            dataset = QNLData(tokenizer=tokenizer,
                              data_path=args.data_path + "QNLI/train.tsv")
        elif data == "SST2":
            dataset = SST2Data(tokenizer=tokenizer,
                               data_path=args.data_path + "SST-2/train.tsv")
        elif data == "AGNews":
            dataset = NewsData(
                tokenizer=tokenizer,
                data_path=args.data_path + "AGNews/train.csv",
            )
        data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=BATCHSIZE)

        result_array_female, result_array_male = male_female_forward_pass(data_loader, tokenizer, BATCHSIZE, device)
        female_male_saving(result_array_male, result_array_female, args.output_path, data)


def gender_debiasing(args):
    male_embeddings, female_embeddings = load_from_database(args.data_path, data_name="QNLI")
    debias = DebiasingPCA(2)
    embeddings = np.concatenate([male_embeddings, female_embeddings])
    label_index = np.concatenate([np.zeros(len(male_embeddings)), np.ones(len(female_embeddings))])
    debias.fit(embeddings, label_index)
    debiased_embeddings = debias.debias(embeddings)


def evaluation_gender_run():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE, device=device)

    dataset = select_data_set(data_name=args.data_name, tokenizer=tokenizer, data_path=args.data_path, mode="train")

    compute_score = ScoreComputer(tokenizer, model, BATCHSIZE, device)

    data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=BATCHSIZE)

    result_array_female, result_array_male = male_female_forward_pass(data_loader, model, BATCHSIZE, device)
    compute_score.read_text_example(result_array_male, result_array_female)
    print("Original score train:", compute_score.compute_score(6, "cosine"), compute_score.compute_score(6, "gaus"))

    debias = DebiasingPCA(3)
    embeddings, label_index = prepare_pca_input(result_array_male, result_array_female)
    debias.fit(embeddings, label_index)
    debiased_embeddings = debias.debias(embeddings)
    compute_score.read_text_example(debiased_embeddings[label_index == 1], debiased_embeddings[label_index == 0])

    print("Debiased score train:",compute_score.compute_score(6, "cosine"), compute_score.compute_score(6, "gaus"))

    dataset = select_data_set(data_name=args.data_name, tokenizer=tokenizer, data_path=args.data_path, mode="test")
    data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=BATCHSIZE)
    result_array_female, result_array_male = male_female_forward_pass(data_loader, model, BATCHSIZE, device)
    compute_score.read_text_example(result_array_male, result_array_female)

    print("Original score test:", compute_score.compute_score(6, "cosine"), compute_score.compute_score(6, "gaus"))

    embeddings, label_index = prepare_pca_input(result_array_male, result_array_female)
    debiased_embeddings = debias.debias(embeddings)
    compute_score.read_text_example(debiased_embeddings[label_index == 1], debiased_embeddings[label_index == 0])

    print("Debiased score test:", compute_score.compute_score(6, "cosine"), compute_score.compute_score(6, "gaus"))


def religion_run():
    # parse arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = QNLDataReligion(tokenizer=tokenizer,
                              data_path="D:/Dokumente/Universitaet/Statistik/ML/NLP_new/debiasing-sent/data"
                                        "/QNLI/dev.tsv")
    data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=BATCHSIZE)
    model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE)
    for el in data_loader.train_loader:
        christ_embedding = model.forward(**el["christ"])
        jew_embedding = model.forward(**el["jew"])
        muslim_embedding = model.forward(**el["muslim"])


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default="./data/", type=str, required=False)
    parser.add_argument('--data-name', default="QNLI", type=str, required=False)
    parser.add_argument('--out-path', default="./data/", type=str, required=False)
    args = parser.parse_args()
    evaluation_gender_run()
