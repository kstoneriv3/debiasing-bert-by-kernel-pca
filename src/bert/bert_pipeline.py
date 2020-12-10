import argparse
import sys
from transformers import BertTokenizer
sys.path.append("/cluster/home/vbardenha/debiasing-sent/")
from src.bert.dataloader import GenericDataLoader, NewsData, QNLData, CoLAData, SST2Data, CoLADataReligion, \
    QNLDataReligion
from src.bert.models import EmbeddingModel, ClassificationHead, ClassificationModel
import torch
import pandas as pd


from src.bert.evaluation import female_male_saving, female_male_dataset_creation, ScoreComputer
from src.bert.dataloader import load_from_database, select_data_set, select_data_set_standard
from src.debiasing.pca import DebiasingPCA
from src.debiasing.torch_kpca import TorchDebiasingKernelPCA
from src.bert.evaluation import male_female_forward_pass, prepare_pca_input, DownstreamPipeline

BATCHSIZE = 8


def gender_example_creation():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE, device=device)
    for mode in ["train", "test"]:
        for data in ["CoLA", "QNLI", "SST2"]:
            if data == "CoLA":
                dataset = CoLAData(tokenizer=tokenizer,
                                   data_path=args.data_path + "CoLA/{}.tsv".format(mode))
            elif data == "QNLI":
                dataset = QNLData(tokenizer=tokenizer,
                                  data_path=args.data_path + "QNLI/{}.tsv".format(mode))
            elif data == "SST2":
                dataset = SST2Data(tokenizer=tokenizer,
                                   data_path=args.data_path + "SST-2/{}.tsv".format(mode))
            elif data == "AGNews":
                dataset = NewsData(
                    tokenizer=tokenizer,
                    data_path=args.data_path + "AGNews/train.csv",
                )
            data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=BATCHSIZE)

            result_array_female, result_array_male = male_female_forward_pass(data_loader, model, BATCHSIZE, device)
            female_male_dataset_creation(result_array_female, result_array_male, args.data_path, data, mode)


def establish_bias_baseline():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE, device=device)

    compute_score = ScoreComputer(tokenizer, model, BATCHSIZE, device)

    result_frame = pd.DataFrame(
        columns=["cosine distance", "cosine p-value", "gaus distance", "gaus p-value", "sigmoid distance",
                "sigmoid p-value"],
        index=["test_6", "test_7", "test_8",
               "test_CoLA_6", "test_CoLA_7", "test_CoLA_8",
               "test_QNLI_6", "test_QNLI_7", "test_QNLI_8",
               "test_SST2_6", "test_SST2_7", "test_SST2_8"])
    for test_type in [6,  7,  8]:
        compute_score.load_original_seat(test_type)
        for distance_metric in ["cosine","gaus","sigmoid"]:
            result_frame.loc["test_{}".format(test_type),"{} distance".format(distance_metric)] = compute_score.compute_score(test_type, distance_metric)
            result_frame.loc["test_{}".format(test_type),"{} p-value".format(distance_metric)] = compute_score.compute_permutation_score(test_type, distance_metric)

            for data_set in ["CoLA", "QNLI", "SST2"]:
                result_array_female, result_array_male = load_from_database(args.data_path, data_set, "train")
                compute_score.read_text_example(result_array_male, result_array_female)

                result_frame.loc["test_{}_{}".format(data_set,test_type), "{} distance".format(distance_metric)] = compute_score.compute_score(test_type,
                                                                                                               distance_metric)

                result_frame.loc["test_{}_{}".format(data_set,test_type), "{} p-value".format(distance_metric)] = compute_score.compute_permutation_score(
                    test_type, distance_metric)
    result_frame.to_latex("./src/experiments/baseline_metric.tex")

def evaluation_gender_run():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE, device=device)

    compute_score = ScoreComputer(tokenizer, model, BATCHSIZE, device)
    result_array_female, result_array_male = load_from_database(args.data_path, args.data_name, "train")
    pd.DataFrame()
    # Compute embeddings for the train dataset

    if args.recompute:
        dataset = select_data_set(data_name=args.data_name, tokenizer=tokenizer, data_path=args.data_path, mode="train")
        data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=BATCHSIZE)
        result_array_female, result_array_male = male_female_forward_pass(data_loader, model, BATCHSIZE, device)
    else:
        result_array_female, result_array_male = load_from_database(args.data_path, args.data_name, "train")
    # Score BERT for the train dataset
    test_type = 8
    compute_score.read_text_example(result_array_male, result_array_female)
    print("My Score train:", compute_score.compute_score(test_type, "cosine"),
          compute_score.compute_score(test_type, "gaus"), compute_score.compute_score(test_type, "sigmoid"))

    print("permutation score p-values:", compute_score.compute_permutation_score(test_type, "cosine"),
          compute_score.compute_permutation_score(test_type, "gaus"),
          compute_score.compute_permutation_score(test_type, "sigmoid")
          )
    # permutation_metric = "gaus"
    #
    # print("permutation difference score p-values:",
    #       compute_score.compute_permutation_difference_score(test_type, permutation_metric))
    # Score BERT for the original examples
    compute_score.load_original_seat()
    print("Original metric train:", compute_score.compute_score(test_type, "cosine"),
          compute_score.compute_score(test_type, "gaus"), compute_score.compute_score(test_type, "sigmoid"))
    print("permutation score p-values:", compute_score.compute_permutation_score(test_type, "cosine"),
          compute_score.compute_permutation_score(test_type, "gaus"),
          compute_score.compute_permutation_score(test_type, "sigmoid")
          )
    # print("permutation difference score p-values:",
    #       compute_score.compute_permutation_difference_score(test_type, permutation_metric))
    # Train the debiasing algorithm
    del model
    torch.cuda.empty_cache()

    debias = DebiasingPCA(2)
    embeddings, label_index = prepare_pca_input(result_array_male, result_array_female)
    debias.fit(embeddings, label_index)

    # Debias train embeddings
    debiased_embeddings = debias.debias(embeddings)

    # Compute scores of debiased embeddings
    compute_score.read_text_example(debiased_embeddings[label_index == 0], debiased_embeddings[label_index == 1])
    print("My score debias train:", compute_score.compute_score(test_type, "cosine"),
          compute_score.compute_score(test_type, "gaus"), compute_score.compute_score(test_type, "sigmoid"))
    print("permutation score p-values:", compute_score.compute_permutation_score(test_type, "cosine"),
          compute_score.compute_permutation_score(test_type, "gaus"),
          compute_score.compute_permutation_score(test_type, "sigmoid")
          )
    # print("permutation difference score p-values:",
    #       compute_score.compute_permutation_difference_score(test_type, permutation_metric))

    # Compute score of original metric after debiasing
    compute_score.load_original_seat()
    compute_score.male_embeddings = debias.debias(compute_score.male_embeddings)
    compute_score.female_embeddings = debias.debias(compute_score.female_embeddings)
    print("original metric debias:", compute_score.compute_score(test_type, "cosine"),
          compute_score.compute_score(test_type, "gaus"), compute_score.compute_score(test_type, "sigmoid"))
    print("permutation score p-values:", compute_score.compute_permutation_score(test_type, "cosine"),
          compute_score.compute_permutation_score(test_type, "gaus"),
          compute_score.compute_permutation_score(test_type, "sigmoid")
          )
    # print("permutation difference score p-values:",
    #       compute_score.compute_permutation_difference_score(test_type, permutation_metric))
    # Analyze results on the test dataset
    model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE, device=device)

    if args.recompute:
        dataset = select_data_set(data_name=args.data_name, tokenizer=tokenizer, data_path=args.data_path, mode="test")
        data_loader = GenericDataLoader(dataset, validation_split=0, batch_size=BATCHSIZE)
        result_array_female, result_array_male = male_female_forward_pass(data_loader, model, BATCHSIZE, device)
    else:
        result_array_female, result_array_male = load_from_database(args.data_path, args.data_name, "test")

    compute_score.read_text_example(result_array_male, result_array_female)
    print("Original score test:", compute_score.compute_score(test_type, "cosine"),
          compute_score.compute_score(test_type, "gaus"), compute_score.compute_score(test_type, "sigmoid"))
    print("permutation score p-values:", compute_score.compute_permutation_score(test_type, "cosine"),
          compute_score.compute_permutation_score(test_type, "gaus"),
          compute_score.compute_permutation_score(test_type, "sigmoid")
          )
    # print("permutation difference score p-values:",
    #       compute_score.compute_permutation_difference_score(test_type, permutation_metric))
    # embeddings, label_index = prepare_pca_input(result_array_male, result_array_female)
    debiased_embeddings = debias.debias(embeddings)
    compute_score.read_text_example(debiased_embeddings[label_index == 0], debiased_embeddings[label_index == 1])
    print("Debiased score test:", compute_score.compute_score(test_type, "cosine"),
          compute_score.compute_score(test_type, "gaus"), compute_score.compute_score(test_type, "sigmoid"))
    print("permutation score p-values:", compute_score.compute_permutation_score(test_type, "cosine"),
          compute_score.compute_permutation_score(test_type, "gaus"),
          compute_score.compute_permutation_score(test_type, "sigmoid")
          )
    # print("permutation difference score p-values:",
    #       compute_score.compute_permutation_difference_score(test_type, permutation_metric))


def downstream_pipeline():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    embedding_model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE, device=device)
    classifier_model = ClassificationHead()
    model = ClassificationModel(embedding_model=embedding_model, classification_model=classifier_model)

    # Compute embeddings for the train dataset
    dataset_train = select_data_set_standard(data_name=args.data_name, tokenizer=tokenizer, data_path=args.data_path,
                                             mode="train")
    dataset_test = select_data_set_standard(data_name=args.data_name, tokenizer=tokenizer, data_path=args.data_path,
                                            mode="dev")

    data_loader = GenericDataLoader(dataset_train, validation_data=dataset_test, batch_size=BATCHSIZE)
    optimizer = torch.optim.Adam(classifier_model.parameters())
    trainer = DownstreamPipeline(model=model, data_loader=data_loader, device=device, optimizer=optimizer,epochs=10)
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
    parser.add_argument('--data-name', default="CoLA", type=str, required=False)
    parser.add_argument('--out-path', default="./data/", type=str, required=False)
    parser.add_argument('--recompute', action="store_true")

    args = parser.parse_args()
    # establish_bias_baseline()
    # gender_example_creation()
    # evaluation_gender_run()
    downstream_pipeline()
