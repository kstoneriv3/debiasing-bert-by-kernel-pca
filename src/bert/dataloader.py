from torch.utils.data import DataLoader, SubsetRandomSampler, dataset
import numpy as np
import csv
import re


# TODO: change to train/test/dev splits as in the given dataset

class GenericDataLoader:
    def __init__(self, dataset,
                 batch_size=1,
                 validation_split=0,
                 shuffle_for_split=True,
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
                self.dataset, batch_size, shuffle=True, drop_last=True)

        # case validation set only (testset)
        if validation_split == 1:
            self.val_loader = DataLoader(
                self.dataset, batch_size, shuffle=True, drop_last=True)

        # case training/validation set split
        else:
            indices = np.arange(len(dataset))
            if shuffle_for_split:
                np.random.seed(random_seed_split)
                indices = np.random.permutation(indices)
            split = int(np.floor(validation_split * len(dataset)))
            train_sampler = SubsetRandomSampler(indices[split:])
            val_sampler = SubsetRandomSampler(indices[:split])
            self.train_loader = DataLoader(
                self.dataset, batch_size, sampler=train_sampler, drop_last=True)
            self.val_loader = DataLoader(
                self.dataset, batch_size, sampler=val_sampler, drop_last=True)


class TokenizeDataset(dataset.Dataset):
    def __init__(self, tokenizer, data_path):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.data = self._read_tsv()

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
        all_pats = ["he", "himself", "boy", "man", "father", "guy", "male", "his", "himself", "John"]
        # TODO: problem with he at the beginning of sentence

        # print(re.findall("\d:\d\d",text))
        pattern = re.compile(r'|'.join(map(r'(?:\s{}\s)'.format, all_pats)))
        found = re.findall(pattern, line)
        if found:
            print(line, found)  # samples are not really convincing yet

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
        self.find_gender_in_text(self.data[item][-1])

        return {
            "label": self.data[item][1],
            "text": self.tokenize_text(self.data[item][-1]),
        }


class QNLData(TokenizeDataset):
    def __init__(self, **kwargs):
        super(QNLData, self).__init__(**kwargs)

    def __getitem__(self, item):
        self.find_gender_in_text(self.data[item][1])
        self.find_gender_in_text(self.data[item][2])

        return {
            "label": self.data[item][-1],
            "text": self.tokenize_text(self.data[item][1]),
            "answer": self.tokenize_text(self.data[item][2]),
        }


class SST2Data(TokenizeDataset):
    def __init__(self, **kwargs):
        super(SST2Data, self).__init__(**kwargs)

    def __getitem__(self, item):
        self.find_gender_in_text(self.data[item][0])

        return {
            "label": self.data[item][-1],
            "text": self.tokenize_text(self.data[item][0]),
        }
