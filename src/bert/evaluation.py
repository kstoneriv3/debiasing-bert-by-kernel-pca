import numpy as np
import h5py
import torch
import json
from sklearn.preprocessing import normalize


def female_male_saving(male_array, female_array, data_path, data_name):
    max_written = np.where(female_array[:, 0] == 0)[0][0]
    result_array_female = np.resize(female_array, (max_written, 768))
    result_array_male = np.resize(male_array, (max_written, 768))

    h5f = h5py.File(data_path + 'data_out.h5', 'r+')
    fem_data = h5f["female_embeddings_{}".format(data_name)]
    fem_data[...] = result_array_female
    male_data = h5f["male_embeddings_{}".format(data_name)]
    male_data[...] = result_array_male
    h5f.close()


def cosine_similarity(emb, el):
    emb = normalize(emb)
    el = normalize(el.reshape(-1, 1), axis=0).reshape(-1)
    return np.matmul(emb, el) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(el))


def rbf_similarity(emb, el, gamma=0.1):
    emb = normalize(emb)
    el = normalize(el.reshape(-1, 1), axis=0).reshape(-1)
    return np.exp(-gamma * np.sum(np.square(emb - el.reshape(1, -1)), axis=1))


def sigmoid_similarity(emb, el, gamma=0.0001, c0=0):
    emb = normalize(emb)
    el = normalize(el.reshape(-1, 1), axis=0).reshape(-1)
    return np.tanh(gamma * np.matmul(emb, el) + c0)


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
        for i in [6, 7, 8]:
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

    def get_text_examples(self, data_name):
        f = h5py.File(self.data_path + '/data_out.h5', 'r')
        female_embeddings = f["female_embeddings_{}".format(data_name)][:]
        male_embeddings = f["male_embeddings_{}".format(data_name)][:]
        return male_embeddings, female_embeddings

    def compute_score(self, i, metric="cosine", data="QNLI"):
        # attribute_1 = self.get_dict_embeddings(i, "attr", 1)
        # attribute_2 = self.get_dict_embeddings(i, "attr", 2)
        attribute_1, attribute_2 = self.get_text_examples(data)
        target_1 = self.get_dict_embeddings(i, "targ", 1)
        target_2 = self.get_dict_embeddings(i, "targ", 2)

        score_target_1 = score(target_1, attribute_1, attribute_2, metric)
        score_target_2 = score(target_2, attribute_1, attribute_2, metric)
        return (np.mean(score_target_1) - np.mean(score_target_2)) / np.std(
            np.concatenate([score_target_1, score_target_2]))

    def compute_all_metrics(self):
        for data in ["SST2", "CoLA", "QNLI"]:
            for i in [6, 7, 8]:
                print(self.compute_score(i, "cosine", data), self.compute_score(i, "gaus", data),
                      self.compute_score(i, "sigmoid", data))
