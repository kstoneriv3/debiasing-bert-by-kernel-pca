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

    def load_text_examples(self, data_name):
        f = h5py.File(self.data_path + '/data_out.h5', 'r')
        female_embeddings = f["female_embeddings_{}".format(data_name)][:]
        male_embeddings = f["male_embeddings_{}".format(data_name)][:]
        self.male_embeddings = male_embeddings
        self.female_embeddings = female_embeddings

    def read_text_example(self, male_embeddings, female_embeddings):
        self.male_embeddings = male_embeddings
        self.female_embeddings = female_embeddings

    def compute_score(self, i, metric="cosine"):
        # attribute_1 = self.get_dict_embeddings(i, "attr", 1)
        # attribute_2 = self.get_dict_embeddings(i, "attr", 2)
        if self.male_embeddings is None:
            raise AttributeError("Embeddings for evaluation not loaded. Run read_text_example or load_text_example "
                                 "first")
        target_1 = self.get_dict_embeddings(i, "targ", 1)
        target_2 = self.get_dict_embeddings(i, "targ", 2)

        score_target_1 = score(target_1, self.male_embeddings, self.female_embeddings, metric)
        score_target_2 = score(target_2, self.male_embeddings, self.female_embeddings, metric)
        return (np.mean(score_target_1) - np.mean(score_target_2)) / np.std(
            np.concatenate([score_target_1, score_target_2]))

    def compute_all_metrics(self):
        for data in ["SST2", "CoLA", "QNLI"]:
            for i in [6, 7, 8]:
                self.load_text_examples(data)
                print(self.compute_score(i, "cosine", data), self.compute_score(i, "gaus", data),
                      self.compute_score(i, "sigmoid", data))


def male_female_forward_pass(data_loader, model, batch_size, device):
    mean_difference = torch.zeros(768, device=device)
    result_array_male = np.empty((len(data_loader.train_loader), 768), dtype=float)
    result_array_female = np.empty((len(data_loader.train_loader), 768), dtype=float)

    try:
        for n, el in enumerate(data_loader.train_loader):
            # TODO: add normalization
            male_embedding = model.forward(**el["male"])[1].detach()
            female_embedding = model.forward(**el["female"])[1].detach()
            result_array_male[n*batch_size:(n +1)* batch_size] = male_embedding.cpu().numpy()
            result_array_female[n*batch_size:(n+1) * batch_size] = female_embedding.cpu().numpy()
            mean_difference = mean_difference * n / (n + 1) + (male_embedding - female_embedding).mean(0) / (n + 1)
    except IndexError:
        # print(mean_difference)
        print(n)
    result_array_female= result_array_female[: np.where(result_array_female[:, 0] == 0)[0][0]]
    result_array_male= result_array_male[: np.where(result_array_male[:, 0] == 0)[0][0]]
    return result_array_female, result_array_male


def prepare_pca_input(result_array_male,result_array_female):
    embeddings = np.concatenate([result_array_male, result_array_female])
    label_index = np.concatenate([np.zeros(len(result_array_male)), np.ones(len(result_array_female))])
    return embeddings, label_index