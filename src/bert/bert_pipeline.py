import argparse
from transformers import BertTokenizer
from src.bert.dataloader import GenericDataLoader, QNLData, CoLAData, SST2Data
from src.bert.models import EmbeddingModel


def main_run():
    # parse arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = CoLAData(tokenizer=tokenizer,
                      data_path="D:/Dokumente/Universitaet/Statistik/ML/NLP_new/debiasing-sent/data"
                                "/CoLA/dev.tsv")
    data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=8)
    model = EmbeddingModel("bert-base-uncased", batch_size=8)
    for el in data_loader.train_loader:
        output = model.forward(**el["text"])


if __name__ == "__main__":
    main_run()
