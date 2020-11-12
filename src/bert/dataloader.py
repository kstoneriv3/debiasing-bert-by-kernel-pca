from torch.utils.data import DataLoader, SubsetRandomSampler, dataset, SequentialSampler
import numpy as np
import csv
import re


class TwoWayDict(dict):
    def __init__(self, input_dict, **kwargs):
        super().__init__(**kwargs)
        for key, value in input_dict.items():
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


# TODO: change to train/test/dev splits as in the given dataset

class GenericDataLoader:
    def __init__(self, dataset,
                 batch_size=1,
                 validation_split=0,
                 shuffle_for_split=False,
                 random_seed_split=0):
        """
        Initializes the dataloader.
        3 configuration are supported:
            *   dataset used for training only (validation_split = 0).
                Only self.train_loader is initialized.
            *   dataset used for validation/testing only (validation_split = 1)
                Only self.val_loader is initialized
            *   dataset to be splitted between validation and training
                Both self.train_loader and self.val_loader are initialized
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # case no validation set
        if validation_split == 0:
            self.train_loader = DataLoader(
                self.dataset, batch_size, shuffle=False, drop_last=True)

        # case validation set only (testset)
        if validation_split == 1:
            self.val_loader = DataLoader(
                self.dataset, batch_size, shuffle=False, drop_last=True)

        # case training/validation set split
        else:
            indices = np.arange(len(dataset))
            if shuffle_for_split:
                np.random.seed(random_seed_split)
                indices = np.random.permutation(indices)
            split = int(np.floor(validation_split * len(dataset)))
            train_sampler = SequentialSampler(indices[split:])
            val_sampler = SequentialSampler(indices[:split])
            self.train_loader = DataLoader(
                self.dataset, batch_size, sampler=train_sampler, drop_last=True)
            self.val_loader = DataLoader(
                self.dataset, batch_size, sampler=val_sampler, drop_last=True)


class TokenizeDataset(dataset.Dataset):
    def __init__(self, tokenizer, data_path):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.item_current = 0
        self.end = 0
        self.data = self._read_tsv()
        all_pats_male = ["he", "himself", "boy", "man", "father", "guy", "male", "his", "himself", "john"]
        all_pats_female = ["she", "herself", "girl", "woman", "mother", "gal", "female", "her", "herself", "mary"]
        self.translation_dict = TwoWayDict(
            {"he": "she", "himself": "herself", "boy": "girl", "man": "woman", "father": "mother", "guy": "gal",
             "male": "female", "his": "her", "himself": "herself", "john": "mary"})

        # TODO: problem with he at the beginning of sentence

        # print(re.findall("\d:\d\d",text))
        self.pattern_male = re.compile(r'|'.join(map(r'(?:\b{}\b)'.format, all_pats_male)))
        self.pattern_female = re.compile(r'|'.join(map(r'(?:\b{}\b)'.format, all_pats_female)))

    def _read_tsv(self):
        tsv_file = open(self.data_path, encoding="utf8")
        tsv_reader = csv.reader(tsv_file, delimiter="\t")
        tsv_reader.__next__()
        lines = []
        for line in tsv_reader:
            lines.append(line)
        return lines

    def __len__(self):
        return len(self.data)

    def find_gender_in_text(self, line):
        found_male = re.findall(self.pattern_male, line)
        found_female = re.findall(self.pattern_female, line)
        if found_male:
            print("male", line, found_male)  # samples are not really convincing yet
        if found_female:
            print("female", line, found_female)

    def replace_gender_in_text(self, line):
        male_version = line
        female_version = line

        found_male = re.findall(self.pattern_male, line)
        found_female = re.findall(self.pattern_female, line)
        if found_male:
            # print(female_version)
            for el in found_male:
                female_version = re.sub(r'(?:\b{}\b)'.format(el), self.translation_dict[el], female_version)
            # print(female_version)
        if found_female:
            # print(male_version)
            for el in found_female:
                male_version = re.sub(r'(?:\b{}\b)'.format(el), self.translation_dict[el], male_version)
            # print(male_version)
        if (not found_male) and (not found_female):
            return line, male_version, female_version, 0
        else:
            return line, male_version, female_version, 1

    def tokenize_text(self, line):
        tokenized_sentence = self.tokenizer(
            line,
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt",
        )
        return tokenized_sentence

    def __getitem__(self, item):
        raise NotImplementedError


class CoLAData(TokenizeDataset):
    def __init__(self, **kwargs):
        super(CoLAData, self).__init__(**kwargs)

    def __getitem__(self, item):
        self.item_current = max(self.item_current, item)
        indicator = 0
        while indicator == 0:
            original, male, female, indicator = self.replace_gender_in_text(self.data[self.item_current][-1].lower())
            self.item_current += 1

        return {
            "label": self.data[item][1],
            "female": self.tokenize_text(male),
            "male": self.tokenize_text(female),
        }


class QNLData(TokenizeDataset):
    def __init__(self, **kwargs):
        super(QNLData, self).__init__(**kwargs)

    def __getitem__(self, item):
        self.item_current = max(self.item_current, item)
        indicator = 0
        while indicator == 0:
            original, male, female, indicator = self.replace_gender_in_text(self.data[self.item_current][1].lower())
            self.item_current += 1

        # TODO: how to handle answer
        # self.replace_gender_in_text(self.data[item][2].lower())

        return {
                   "label": self.data[item][-1],
                   "male": self.tokenize_text(male),
                   "female": self.tokenize_text(female),
               }


class SST2Data(TokenizeDataset):
    def __init__(self, **kwargs):
        super(SST2Data, self).__init__(**kwargs)

    def __getitem__(self, item):
        self.item_current = max(self.item_current, item)
        indicator = 0
        while indicator == 0:
            original, male, female, indicator = self.replace_gender_in_text(self.data[self.item_current][0].lower())
            self.item_current += 1
        return {
            "label": self.data[item][-1],
            "male": self.tokenize_text(male),
            "female": self.tokenize_text(female)
        }
