import numpy as np
import h5py
import torch
import json


def female_male_saving(male_array, female_array, data_path, data_name):
    max_written = np.where(female_array[:, 0] == 0)[0][0]
    result_array_female = np.resize(female_array, (max_written, 768))
    result_array_male = np.resize(male_array, (max_written, 768))

    h5f = h5py.File(data_path + 'data_out.h5', 'w')
    h5f.create_dataset("female_embeddings_{}".format(data_name), data=result_array_female)
    h5f.create_dataset("male_embeddings_{}".format(data_name), data=result_array_male)
    h5f.close()


def score(emb, attribute_list_a, attribute_list_b):
    res_a = np.zeros(len(emb))
    res_b = np.zeros(len(emb))
    for el in attribute_list_a:
        res_a += np.matmul(emb, el) / (np.linalg.norm(emb,axis=1) * np.linalg.norm(el))
        if np.any(np.isnan(res_a)):
            print("alarm")
    for el in attribute_list_b:
        res_b += np.matmul(emb, el) / (np.linalg.norm(emb,axis=1) * np.linalg.norm(el))
        if np.any(np.isnan(res_b)):
            print("alarm")
    return res_a / len(attribute_list_a) - res_b / len(attribute_list_b)


class ScoreComputer:
    def __init__(self, tokenizer,model,batch_size,device,tokenizer_max_length=50):
        self.tokenizer = tokenizer
        self.batch_size= batch_size
        self.main_dict = {}
        self.read_json()
        self.model=model
        self.tokenizer_max_length=tokenizer_max_length
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

    def get_dict_embeddings(self,i,category,category_id):
        embedding = np.empty((len(self.main_dict[i]["{}{}".format(category, category_id)]["examples"]),768))

        for j in range(0, len(self.main_dict[i]["{}{}".format(category, category_id)]["examples"]), self.batch_size):
            tokenized = self.tokenize_text(self.main_dict[i]["{}{}".format(category, category_id)]["examples"][j:j + self.batch_size])
            embedding[j:j + self.batch_size] = self.model.forward(**tokenized)[1].detach().cpu().numpy()
        return embedding

    def compute_score(self, i):
        attribute_1 = self.get_dict_embeddings(i,"attr",1)
        attribute_2 = self.get_dict_embeddings(i,"attr",2)
        target_1 = self.get_dict_embeddings(i,"targ",1)
        target_2 = self.get_dict_embeddings(i,"targ",2)

        score_target_1 = score(target_1,attribute_1,attribute_2)
        score_target_2 = score(target_2,attribute_1,attribute_2)
        return (np.mean(score_target_1)-np.mean(score_target_2))/np.std(np.concatenate([score_target_1,score_target_2]))
