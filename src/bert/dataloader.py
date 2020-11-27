from torch.utils.data import DataLoader, SubsetRandomSampler, dataset, SequentialSampler
from torch.utils.data import Sampler
import numpy as np
import csv
import re
import json


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
    def __init__(self, tokenizer, data_path, tokenizer_max_length=50):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.item_current = 0
        self.end = 0
        self.data = self._read_tsv()
        self.tokenizer_max_length = tokenizer_max_length
        all_pats_male = ["he", "himself", "boy", "man", "father", "guy", "male", "his", "himself", "john"]
        all_pats_female = ["she", "herself", "girl", "woman", "mother", "gal", "female", "her", "herself", "mary"]
        self.translation_dict = TwoWayDict(dict(zip(all_pats_male, all_pats_female)))
        all_pats_christ = ["christian", "christians", "bible", "church", "imam"]
        all_pats_muslim = ["muslim", "muslims", "quran", "mosque", "priest"]
        all_pats_jew = ["jewish", "jews", "torah", "synagogue", "rabbi"]
        self.christ_jew = TwoWayDict(dict(zip(all_pats_christ, all_pats_jew)))
        self.christ_muslim = TwoWayDict(dict(zip(all_pats_christ, all_pats_muslim)))
        self.muslim_jew = TwoWayDict(dict(zip(all_pats_jew, all_pats_muslim)))

        self.pattern_male = re.compile(r'|'.join(map(r'(?:\b{}\b)'.format, all_pats_male)))
        self.pattern_female = re.compile(r'|'.join(map(r'(?:\b{}\b)'.format, all_pats_female)))

        self.pattern_christ = re.compile(r'|'.join(map(r'(?:\b{}\b)'.format, all_pats_christ)))
        self.pattern_muslim = re.compile(r'|'.join(map(r'(?:\b{}\b)'.format, all_pats_muslim)))
        self.pattern_jew = re.compile(r'|'.join(map(r'(?:\b{}\b)'.format, all_pats_jew)))

        all_pats_career = ["careers""career","businesses","business","executive", "management", "professional",
                           "corporation", "salary", "salaries", "office", "offices"]
        all_pats_home = ["relative", "relatives", "home", "parent", "parents", "child", "children", "family", "cousin",
                         "cousins", "marriage", "marriages", "weddings"]

        self.pattern_career = re.compile(r'|'.join(map(r'(?:\b{}\b)'.format, all_pats_career)))
        self.pattern_home = re.compile(r'|'.join(map(r'(?:\b{}\b)'.format, all_pats_home)))
        # TODO: arbitrary removed numbers
        all_pats_math = ["math","algebra","geometry","calculus","equation", "computation"]
                           # "addition", "additions"]
        all_pats_arts = ["poetry", "art", "dance", "dances", "literature", "novel", "novels", "symphony", "symphonies",
                         "drama", "dramas", "sculpture", "sculptures"]

        self.pattern_math = re.compile(r'|'.join(map(r'(?:\b{}\b)'.format, all_pats_math)))
        self.pattern_arts = re.compile(r'|'.join(map(r'(?:\b{}\b)'.format, all_pats_arts)))

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

    def find_career_in_text(self, line):
        found_career = re.findall(self.pattern_career, line)
        found_home = re.findall(self.pattern_home, line)
        if found_career:
            print("male", line, found_career)  # samples are not really convincing yet
        if found_home:
            print("female", line, found_home)

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

    def replace_religion_in_text(self, line):
        christian_version = line
        muslim_version = line
        jewish_version = line

        found_christ = re.findall(self.pattern_christ, line)
        found_muslim = re.findall(self.pattern_muslim, line)
        found_jew = re.findall(self.pattern_jew, line)

        if found_christ:
            # print(female_version)
            for el in found_christ:
                muslim_version = re.sub(r'(?:\b{}\b)'.format(el), self.christ_muslim[el], muslim_version)
                jewish_version = re.sub(r'(?:\b{}\b)'.format(el), self.christ_jew[el], jewish_version)

            # print(female_version)
        if found_jew:
            # print(female_version)
            for el in found_jew:
                muslim_version = re.sub(r'(?:\b{}\b)'.format(el), self.muslim_jew[el], muslim_version)
                christian_version = re.sub(r'(?:\b{}\b)'.format(el), self.christ_jew[el], christian_version)
            # print(female_version)
        if found_muslim:
            # print(male_version)
            for el in found_muslim:
                jewish_version = re.sub(r'(?:\b{}\b)'.format(el), self.muslim_jew[el], jewish_version)
                christian_version = re.sub(r'(?:\b{}\b)'.format(el), self.christ_muslim[el], christian_version)
                # print(male_version)
        if (not found_christ) and (not found_jew) and (not found_muslim):
            return line, christian_version, muslim_version, jewish_version, 0
        else:
            return line, christian_version, muslim_version, jewish_version, 1

    def tokenize_text(self, line):
        tokenized_sentence = self.tokenizer(
            line,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
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
            # self.find_career_in_text(self.data[self.item_current][-1].lower())
            self.item_current += 1

        return {
            "label": self.data[item][1],
            "female": self.tokenize_text(female),
            "male": self.tokenize_text(male),
        }


class NewsData(TokenizeDataset):
    def __init__(self, **kwargs):
        super(NewsData, self).__init__(**kwargs)

    def _read_tsv(self):
        tsv_file = open(self.data_path, encoding="utf8")
        tsv_reader = csv.reader(tsv_file, delimiter=",")
        tsv_reader.__next__()
        lines = []
        for line in tsv_reader:
            text = line[2]
            text = text[text.find("-")+1:]
            text = text.split(".")[0]
            line[2]=re.sub(r"[^a-zA-Z0-9,.]+", ' ',text)
            lines.append(line)
        return lines

    def __getitem__(self, item):
        self.item_current = max(self.item_current, item)
        indicator = 0
        while indicator == 0:
            original, male, female, indicator = self.replace_gender_in_text(self.data[self.item_current][-1].lower())
            # self.find_career_in_text(self.data[self.item_current][-1].lower())
            self.item_current += 1

        return {
            "label": self.data[item][1],
            "female": self.tokenize_text(female),
            "male": self.tokenize_text(male),
        }

class QNLData(TokenizeDataset):
    def __init__(self, **kwargs):
        super(QNLData, self).__init__(**kwargs)

    def __getitem__(self, item):
        self.item_current = max(self.item_current, item)
        indicator = 0
        while indicator == 0:
            original, male, female, indicator = self.replace_gender_in_text(self.data[self.item_current][1].lower())
            # self.find_career_in_text(self.data[self.item_current][-1].lower())
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
            # self.find_career_in_text(self.data[self.item_current][-1].lower())
            self.item_current += 1
        return {
            "label": self.data[item][-1],
            "male": self.tokenize_text(male),
            "female": self.tokenize_text(female)
        }


class CoLADataReligion(TokenizeDataset):
    def __init__(self, **kwargs):
        super(CoLADataReligion, self).__init__(**kwargs)

    def __getitem__(self, item):
        self.item_current = max(self.item_current, item)
        indicator = 0
        while indicator == 0:
            original, christ, muslim, jew, indicator = self.replace_religion_in_text(
                self.data[self.item_current][-1].lower())
            self.item_current += 1
        print(christ, muslim)
        return {
            "label": self.data[item][1],
            "christ": self.tokenize_text(christ),
            "jew": self.tokenize_text(jew),
            "muslim": self.tokenize_text(muslim),
        }


class SST2DataReligion(TokenizeDataset):
    def __init__(self, **kwargs):
        super(SST2DataReligion, self).__init__(**kwargs)

    def __getitem__(self, item):
        self.item_current = max(self.item_current, item)
        indicator = 0
        while indicator == 0:
            original, christ, jew, muslim, indicator = self.replace_religion_in_text(
                self.data[self.item_current][0].lower())
            self.item_current += 1
        print(christ, jew)
        return {
            "label": self.data[item][1],
            "christ": self.tokenize_text(christ),
            "jew": self.tokenize_text(jew),
            "muslim": self.tokenize_text(muslim),
        }


class QNLDataReligion(TokenizeDataset):
    def __init__(self, **kwargs):
        super(QNLDataReligion, self).__init__(**kwargs)

    def __getitem__(self, item):
        self.item_current = max(self.item_current, item)
        indicator = 0
        while indicator == 0:
            original, christ, muslim, jew, indicator = self.replace_religion_in_text(
                self.data[self.item_current][1].lower())
            self.item_current += 1

        # TODO: how to handle answer
        # self.replace_gender_in_text(self.data[item][2].lower())
        print(christ, jew)

        return {
            "label": self.data[item][-1],
            "christ": self.tokenize_text(christ),
            "muslim": self.tokenize_text(muslim),
            "jew": self.tokenize_text(jew),
        }
