import h5py
import json
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import scipy.stats as stats


def female_male_saving(male_array, female_array, data_path, data_name, mode="train"):
    h5f = h5py.File(data_path + 'data_out.h5', 'r+')
    fem_data = h5f["female_embeddings_{}_{}".format(data_name, mode)]
    fem_data[...] = female_array
    male_data = h5f["male_embeddings_{}_{}".format(data_name, mode)]
    male_data[...] = male_array
    h5f.close()


def female_male_dataset_creation(male_array, female_array, data_path, data_name, mode):
    h5f = h5py.File(data_path + 'data_out.h5', 'a')
    h5f.create_dataset("female_embeddings_{}_{}".format(data_name, mode), data=female_array)
    h5f.create_dataset("male_embeddings_{}_{}".format(data_name, mode), data=male_array)
    h5f.close()


def cosine_similarity(emb, el):
    # emb = normalize(emb)
    # el = normalize(el.reshape(-1, 1), axis=0).reshape(-1)
    return np.matmul(emb, el) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(el))


def full_cosine_similarity(emb, attribute_list):
    return np.divide(np.matmul(emb, np.transpose(attribute_list)),
                     (np.linalg.norm(emb, axis=1) * np.linalg.norm(attribute_list)).reshape(-1, 1))


def rbf_similarity(emb, el, gamma=0.024):
    # emb = normalize(emb)
    # el = normalize(el.reshape(-1, 1), axis=0).reshape(-1)
    return np.exp(-gamma * np.sum(np.square(emb - el.reshape(1, -1)), axis=1))


def full_rbf_similarity(emb, attribute_list, gamma=0.024):
    return np.exp(-gamma * euclidean_distances(emb, attribute_list, squared=True))


def sigmoid_similarity(emb, el, gamma=0.0001, c0=0):
    emb = normalize(emb)
    el = normalize(el.reshape(-1, 1), axis=0).reshape(-1)
    return np.tanh(gamma * np.matmul(emb, el) + c0)


def full_sigmoid_similarity(emb, attribute_list, gamma=1 / 40, c0=-274):
    return np.tanh(gamma * (np.matmul(emb, np.transpose(attribute_list)) + c0))


def full_score(emb, attribute_list_a, attribute_list_b, metric):
    if metric == "cosine":
        similarity = full_cosine_similarity
    elif metric == "gaus":
        similarity = full_rbf_similarity
    elif metric == "sigmoid":
        similarity = full_sigmoid_similarity
    else:
        raise NotImplementedError
    sim_score_a = np.mean(similarity(emb, attribute_list_a), axis=1)
    sim_score_b = np.mean(similarity(emb, attribute_list_b), axis=1)
    return sim_score_a - sim_score_b


def score(emb, attribute_list_a, attribute_list_b, metric):
    if metric == "cosine":
        similarity = cosine_similarity
    elif metric == "gaus":
        similarity = rbf_similarity
    elif metric == "sigmoid":
        similarity = sigmoid_similarity

    res_a = np.zeros(len(emb))
    res_b = np.zeros(len(emb))
    for el in attribute_list_a:
        res_a += similarity(emb, el)
        assert not np.any(np.isnan(res_a))
    for el in attribute_list_b:
        res_b += similarity(emb, el)
        assert not np.any(np.isnan(res_b))
    return res_a / len(attribute_list_a) - res_b / len(attribute_list_b)


class ScoreComputer:
    def __init__(self, tokenizer, model, batch_size, device, data_path="./data", tokenizer_max_length=50):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_path = data_path
        self.main_dict = {}
        self.read_json()
        self.model = model
        self.tokenizer_max_length = tokenizer_max_length
        self.model.to(device)
        self.model.eval()
        self.male_embeddings = None
        self.female_embeddings = None

    def tokenize_text(self, line):
        tokenized_sentence = self.tokenizer(
            line,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        return tokenized_sentence

    def read_json(self):
        for i in [6, "6b", 7, "7b", 8, "8b"]:
            with open('./src/seat/sent-weat{}.jsonl'.format(i), ) as json_file:
                self.main_dict[i] = json.load(json_file)
        print(self.main_dict)

    def get_dict_embeddings(self, i, category, category_id):
        embedding = np.empty((len(self.main_dict[i]["{}{}".format(category, category_id)]["examples"]), 768))

        for j in range(0, len(self.main_dict[i]["{}{}".format(category, category_id)]["examples"]), self.batch_size):
            tokenized = self.tokenize_text(
                self.main_dict[i]["{}{}".format(category, category_id)]["examples"][j:j + self.batch_size])
            embedding[j:j + self.batch_size] = self.model.forward(**tokenized)[1].detach().cpu().numpy()
        return embedding

    def load_text_examples(self, data_name):
        f = h5py.File(self.data_path + '/data_out.h5', 'r')
        female_embeddings = f["female_embeddings_{}".format(data_name)][:]
        male_embeddings = f["male_embeddings_{}".format(data_name)][:]
        self.male_embeddings = male_embeddings
        self.female_embeddings = female_embeddings

    def read_text_example(self, male_embeddings, female_embeddings):
        self.male_embeddings = male_embeddings
        self.female_embeddings = female_embeddings

    def load_original_seat(self, test):
        self.male_embeddings = self.get_dict_embeddings(test, "targ", 1)
        self.female_embeddings = self.get_dict_embeddings(test, "targ", 2)

    def compute_score(self, i, metric="cosine"):
        if self.male_embeddings is None:
            raise AttributeError("Embeddings for evaluation not loaded. Run read_text_example or load_text_example "
                                 "first")
        target_1 = self.get_dict_embeddings(i, "attr", 1)
        target_2 = self.get_dict_embeddings(i, "attr", 2)

        score_target_1 = full_score(self.male_embeddings, target_1, target_2, metric)
        score_target_2 = full_score(self.female_embeddings, target_1, target_2, metric)
        return (np.mean(score_target_1) - np.mean(score_target_2))

    # def compute_permutation_difference_score(self, i, metric):
    #     target_1 = self.get_dict_embeddings(i, "attr", 1)
    #     target_2 = self.get_dict_embeddings(i, "attr", 2)
    #     score_full = full_score(self.female_embeddings - self.male_embeddings, target_1, target_2, metric)
    #     permute = np.random.choice((-1, 1), len(score_full) * 10000, replace=True).reshape(-1, 10000)
    #     score_full_permute = np.multiply(np.transpose(permute), score_full)
    #     score_full_permute = np.mean(score_full_permute, axis=1)
    #     score_real = np.mean(score_full)
    #     return stats.percentileofscore(score_full_permute, score_real) / 100

    def compute_permutation_difference_score(self, i, metric):
        target_1 = self.get_dict_embeddings(i, "attr", 1)
        target_2 = self.get_dict_embeddings(i, "attr", 2)
        score_real = np.nanmean(full_score(self.female_embeddings - self.male_embeddings, target_1, target_2, metric))
        score_permute = np.zeros(1000)
        score_permute[0] = score_real
        for i in range(1, 1000):
            permute = np.random.choice((-1, 1), len(self.female_embeddings), replace=True)
            tmp_female = np.transpose(np.multiply(np.transpose(self.female_embeddings), permute))
            tmp_male = np.transpose(np.multiply(np.transpose(self.male_embeddings), -permute))
            score_permute[i] = np.nanmean(full_score(tmp_male - tmp_female, target_1, target_2, metric))

        np.sum(score_permute >= score_real) / (len(score_permute))
        np.sum(score_permute <= score_real) / (len(score_permute))

        return min(np.sum(score_permute >= score_real) / len(score_permute),
                   np.sum(score_permute <= score_real) / len(score_permute))

    def compute_permutation_score(self, i, metric="cosine"):
        target_1 = self.get_dict_embeddings(i, "attr", 1)
        target_2 = self.get_dict_embeddings(i, "attr", 2)

        score_target_1 = full_score(self.male_embeddings, target_1, target_2, metric)
        score_target_2 = full_score(self.female_embeddings, target_1, target_2, metric)
        permute = np.random.choice((-1, 1), len(score_target_1) * 10000, replace=True).reshape(-1, 10000)
        score_target_1_permute = np.multiply(np.transpose(permute), score_target_1)
        score_target_2_permute = np.multiply(np.transpose(permute), score_target_2)
        score_real = (np.mean(score_target_1) - np.mean(score_target_2))
        score_dist = np.mean(score_target_1_permute + score_target_2_permute, axis=1)
        return 1 - stats.percentileofscore(score_dist, score_real) / 100

    def compute_all_metrics(self):
        for data in ["SST2", "CoLA", "QNLI"]:
            for i in [6, 7, 8]:
                self.load_text_examples(data)
                print(self.compute_score(i, "cosine", data), self.compute_score(i, "gaus", data),
                      self.compute_score(i, "sigmoid", data))


def male_female_forward_pass(data_loader, model, batch_size, device):
    result_array_male = np.empty((len(data_loader.train_loader) * batch_size, 768), dtype=float)
    result_array_female = np.empty((len(data_loader.train_loader) * batch_size, 768), dtype=float)

    try:
        for n, el in enumerate(data_loader.train_loader):
            # TODO: add normalization
            male_embedding = model.forward(**el["male"])[1].detach().cpu().numpy()
            female_embedding = model.forward(**el["female"])[1].detach().cpu().numpy()
            result_array_male[n * batch_size:(n + 1) * batch_size] = male_embedding
            result_array_female[n * batch_size:(n + 1) * batch_size] = female_embedding
    except IndexError or ValueError:
        # print(mean_difference)
        print("finish")

    del el
    torch.cuda.empty_cache()

    result_array_female = result_array_female[: np.where(result_array_female[:, 0] == 0)[0][0]]
    result_array_male = result_array_male[: np.where(result_array_male[:, 0] == 0)[0][0]]
    return result_array_female, result_array_male


def prepare_pca_input(result_array_male, result_array_female):
    embeddings = np.zeros((result_array_male.shape[0] + result_array_female.shape[0], result_array_male.shape[1]))
    embeddings[::2] = result_array_male
    embeddings[1::2] = result_array_female
    label_index = np.repeat(np.arange(len(result_array_male)), 2)
    return embeddings, label_index


class DownstreamPipeline:
    def __init__(self, data_loader, model, device, optimizer, epochs=100, experiment_name="set_up"):
        self.train_loader = data_loader.train_loader
        self.val_loader = data_loader.val_loader
        self.model = model
        self.device = device
        self.model.to(device)
        self.epochs = epochs
        self.optimizer = optimizer
        time_str = time.strftime("-%Y%m%d%H%M%S")
        self.writer = SummaryWriter(os.path.join("./src/experiments/logs/",
                                                 experiment_name + time_str))

        self.total = None
        self.correct = None

    def train_step(self, batch_data):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(**batch_data["data"])
        batch_data["label"] = batch_data["label"].to(self.device)
        if self.model.classification_model.n_classes == 1:
            loss = nn.BCELoss()(out, batch_data["label"].float().reshape(-1, 1))
        else:
            loss = nn.CrossEntropyLoss()(out, batch_data["label"].float().reshape(-1, 1))
        loss.backward()
        self.optimizer.step()
        return loss

    def val_step(self, batch_data):
        self.model.eval()
        with torch.no_grad:
            output = self.model(**batch_data["data"])
            batch_data["label"] = batch_data["label"].to(self.device)

            self.total += batch_data["labels"].size(0)
            self.correct += self.torch.sum(output == batch_data["label"].float().reshape(-1, 1))

    def train(self):
        for epoch in range(self.epochs):
            for batch_id, batch_data in enumerate(self.train_loader):
                loss = self.train_step(batch_data)
                self.writer.add_scalar(
                    'loss/training',
                    loss,
                    epoch * len(self.train_loader) + batch_id
                )
            self.total = 0
            self.correct = 0
            for batch_id, batch_data in enumerate(self.val_loader):
                self.val_step(batch_data)
                self.writer.add_scalar(
                    'val/accuracy',
                    self.correct / self.total,
                    epoch * len(self.val_loader) + batch_id
                )
