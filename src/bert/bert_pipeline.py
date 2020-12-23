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
from src.debiasing.mixed_kpca import MixedDebiasingKernelPCA
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


def create_debiased_dataset():
    # optim_params = dict(n_iter=30, lr=0.4, alpha=0.)
    for data_name in ["CoLA", "QNLI"]:
        # SST2 has not enough test examples
        result_array_female, result_array_male = load_from_database(args.data_path, data_name, "train")
        debias = MixedDebiasingKernelPCA(2, kernel="rbf", gamma=0.024)
        embeddings, label_index = prepare_pca_input(result_array_male, result_array_female)

        debias.fit(embeddings, label_index)
        result_array_female, result_array_male = load_from_database(args.data_path, data_name, "test")
        embeddings_test, label_index_test = prepare_pca_input(result_array_male,
                                                              result_array_female)
        embeddings_debiased = debias.debias(embeddings_test)
        print("debias successfull")
        debiased_male = embeddings_debiased[::2]
        debiased_female = embeddings_debiased[1::2]
        female_male_dataset_creation(debiased_male, debiased_female, data_path=args.data_path, data_name=data_name,
                                     mode="test_debias")


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

    for test_type in [6, 7, 8]:
        # For all association tests
        for distance_metric in ["cosine", "gaus", "sigmoid"]:
            # For all distance measurements
            compute_score.load_original_seat(test_type)
            test_name = "test_{}".format(test_type),

            result_frame.loc[test_name, "{} distance".format(distance_metric)] = compute_score.compute_score(
                test_type, distance_metric)
            result_frame.loc[test_name, "{} p-value".format(
                distance_metric)] = compute_score.compute_permutation_score(test_type, distance_metric)

            for data_set in ["CoLA", "QNLI", "SST2"]:
                # for all datasets
                result_array_female, result_array_male = load_from_database(args.data_path, data_set, "train")
                compute_score.read_text_example(result_array_male, result_array_female)
                test_name = "test_{}_{}".format(data_set, test_type)

                result_frame.loc[test_name, "{} distance".format(
                    distance_metric)] = compute_score.compute_score(test_type,
                                                                    distance_metric)
                result_frame.loc[test_name, "{} p-value".format(
                    distance_metric)] = compute_score.compute_permutation_score(
                    test_type, distance_metric)

                if not data_set == "SST2":
                    # in SST2 test set there are not enough gendered examples in the test set found
                    debiased_female, debiased_male = load_from_database(args.data_path, data_set, "test_debias")

                    compute_score.read_text_example(debiased_male, debiased_female)
                    test_name = test_name + "_debias"
                    result_frame.loc[test_name, "{} distance".format(
                        distance_metric)] = compute_score.compute_score(test_type,
                                                                        distance_metric)

                    result_frame.loc[test_name, "{} p-value".format(
                        distance_metric)] = compute_score.compute_permutation_score(
                        test_type, distance_metric)

    result_frame.to_latex("./src/experiments/baseline_metric_no_debias.tex")


def downstream_pipeline():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    embedding_model = EmbeddingModel("bert-base-uncased", batch_size=BATCHSIZE, device=device)
    classifier_model = ClassificationHead()

    if args.debias_method == "pca":
        debias = DebiasingPCA(2)
    elif args.debias_method == "kpca":
        debias = MixedDebiasingKernelPCA(2)
    else:
        debias = None

    if (args.debias_method == "pca") or (args.debias_method == "kpca"):
        result_array_female, result_array_male = load_from_database(args.data_path, args.data_name, "train")
        embeddings, label_index = prepare_pca_input(result_array_male, result_array_female)
        debias.fit(embeddings, label_index)
        debias.transfer_to_torch(device)

    model = ClassificationModel(embedding_model=embedding_model,
                                classification_model=classifier_model,
                                do_debiasing=(args.debias_method == "pca") or (args.debias_method == "kpca"),
                                debiasing_model=debias)

    # Compute embeddings for the train dataset
    dataset_train = select_data_set_standard(data_name=args.data_name, tokenizer=tokenizer, data_path=args.data_path,
                                             mode="train")
    dataset_test = select_data_set_standard(data_name=args.data_name, tokenizer=tokenizer, data_path=args.data_path,
                                            mode="dev")

    data_loader = GenericDataLoader(dataset_train, validation_data=dataset_test, batch_size=BATCHSIZE)
    optimizer = torch.optim.Adam(classifier_model.parameters())
    trainer = DownstreamPipeline(model=model, data_loader=data_loader, device=device, optimizer=optimizer, epochs=10)
    trainer.train()


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default="./data/", type=str, required=False)
    parser.add_argument('--data-name', default="CoLA", type=str, required=False)
    parser.add_argument('--out-path', default="./data/", type=str, required=False)
    parser.add_argument('--recompute', action="store_true")
    parser.add_argument('--debias-method', default="none",type=str,required=False)

    args = parser.parse_args()

    # Run
    #   1. Download data by running src/experiments/download_data.py
    #   2. Create Embeddings for sentences that have a gender dimension
    gender_example_creation()
    # #   3. Create the dataset after applying debiasing approaches to gendered sentences
    create_debiased_dataset()
    # #   4. Evaluate SEAT before and after Debiasing was applied
    establish_bias_baseline()
    #   5. Compute downstream performance with debiasing or without debiasing
    downstream_pipeline()
