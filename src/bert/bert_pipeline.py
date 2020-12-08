import argparse
from transformers import BertTokenizer
from src.bert.dataloader import GenericDataLoader, NewsData, QNLData, CoLAData, SST2Data, CoLADataReligion, \
    QNLDataReligion
from src.bert.models import EmbeddingModel, ClassificationHead,ClassificationModel
import torch

from src.bert.evaluation import female_male_saving, ScoreComputer
from src.bert.dataloader import load_from_database, select_data_set, select_data_set_standard
from src.debiasing.pca import DebiasingPCA
from src.debiasing.torch_kpca import TorchDebiasingKernelPCA
from src.bert.evaluation import male_female_forward_pass, prepare_pca_input, DownstreamPipeline

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


def evaluation_gender_run():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE, device=device)
    compute_score = ScoreComputer(tokenizer, model, BATCHSIZE, device)

    # Compute embeddings for the train dataset
    dataset = select_data_set(data_name=args.data_name, tokenizer=tokenizer, data_path=args.data_path, mode="train")
    data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=BATCHSIZE)
    result_array_female, result_array_male = male_female_forward_pass(data_loader, model, BATCHSIZE, device)

    # Score BERT for the train dataset
    compute_score.read_text_example(result_array_male, result_array_female)
    print("My Score train:", compute_score.compute_score(7, "cosine"), compute_score.compute_score(7, "gaus"))

    # Score BERT for the original examples
    compute_score.load_original_seat()
    print("Original metric train:", compute_score.compute_score(7, "cosine"),
          compute_score.compute_score(7, "gaus"))

    # Train the debiasing algorithm
    del model
    torch.cuda.empty_cache()

    debias = TorchDebiasingKernelPCA(5)
    embeddings, label_index = prepare_pca_input(result_array_male[:100], result_array_female[:100])
    debias.fit(embeddings, label_index)

    # Debias train embeddings
    debiased_embeddings = debias.debias(embeddings)

    # Compute scores of debiased embeddings
    compute_score.read_text_example(debiased_embeddings[label_index == 0], debiased_embeddings[label_index == 1])
    print("My score debias train:", compute_score.compute_score(7, "cosine"), compute_score.compute_score(7, "gaus"))

    # Compute score of original metric after debiasing
    compute_score.load_original_seat()
    compute_score.male_embeddings = debias.debias(compute_score.male_embeddings)
    compute_score.female_embeddings = debias.debias(compute_score.female_embeddings)
    print("original metric debias:", compute_score.compute_score(7, "cosine"),
          compute_score.compute_score(7, "gaus"))

    # Analyze results on the test dataset
    model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE, device=device)
    dataset = select_data_set(data_name=args.data_name, tokenizer=tokenizer, data_path=args.data_path, mode="test")
    data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=BATCHSIZE)

    result_array_female, result_array_male = male_female_forward_pass(data_loader, model, BATCHSIZE, device)

    compute_score.read_text_example(result_array_male, result_array_female)
    print("Original score test:", compute_score.compute_score(7, "cosine"), compute_score.compute_score(7, "gaus"))

    embeddings, label_index = prepare_pca_input(result_array_male, result_array_female)
    debiased_embeddings = debias.debias(embeddings)
    compute_score.read_text_example(debiased_embeddings[label_index == 0], debiased_embeddings[label_index == 1])
    print("Debiased score test:", compute_score.compute_score(7, "cosine"), compute_score.compute_score(7, "gaus"))


def downstream_pipeline():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    embedding_model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE, device=device)
    classifier_model = ClassificationHead()
    model = ClassificationModel(embedding_model=embedding_model,classification_model=classifier_model)

    # Compute embeddings for the train dataset
    dataset_train = select_data_set_standard(data_name=args.data_name, tokenizer=tokenizer, data_path=args.data_path, mode="train")
    dataset_test = select_data_set_standard(data_name=args.data_name, tokenizer=tokenizer, data_path=args.data_path, mode="test")

    data_loader = GenericDataLoader(dataset_train, validation_data=dataset_test, batch_size=BATCHSIZE)
    optimizer = torch.optim.Adam(classifier_model.parameters())
    trainer = DownstreamPipeline(model=model,data_loader=data_loader,device=device,optimizer=optimizer)
    trainer.train()

# def religion_run():
#     # parse arguments
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     dataset = QNLDataReligion(tokenizer=tokenizer,
#                               data_path="D:/Dokumente/Universitaet/Statistik/ML/NLP_new/debiasing-sent/data"
#                                         "/QNLI/dev.tsv")
#     data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=BATCHSIZE)
#     model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE)
#     for el in data_loader.train_loader:
#         christ_embedding = model.forward(**el["christ"])
#         jew_embedding = model.forward(**el["jew"])
#         muslim_embedding = model.forward(**el["muslim"])


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default="./data/", type=str, required=False)
    parser.add_argument('--data-name', default="QNLI", type=str, required=False)
    parser.add_argument('--out-path', default="./data/", type=str, required=False)
    args = parser.parse_args()
    # evaluation_gender_run()
    downstream_pipeline()
