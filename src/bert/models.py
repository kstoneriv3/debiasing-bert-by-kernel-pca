from transformers import BertModel
import torch.nn as nn
import torch


class EmbeddingModel(nn.Module):
    def __init__(self, model_name, batch_size, device, tokenizer_max_length=50):
        super(EmbeddingModel, self).__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        self.embedding_model = BertModel.from_pretrained("bert-base-uncased")
        self.embedding_model.to(self.device)
        self.embedding_model.eval()
        self.tokenizer_max_length = tokenizer_max_length

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.embedding_model.forward(input_ids=input_ids.reshape(-1, self.tokenizer_max_length).to(self.device),
                                            attention_mask=attention_mask.reshape(-1, self.tokenizer_max_length).to(
                                                self.device),
                                            token_type_ids=token_type_ids.reshape(-1, self.tokenizer_max_length).to(
                                                self.device))


class ClassificationHead(nn.Module):
    def __init__(self, dense_dim=500, n_classes=1):
        super(ClassificationHead, self).__init__()
        self.dense_dim = dense_dim
        self.n_classes = n_classes
        self.lin = nn.Sequential(
            nn.Linear(768, self.dense_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.dense_dim, self.n_classes),
        )
        if self.n_classes == 1:
            self.head = nn.Sigmoid()
        else:
            self.head = nn.Softmax()

    def forward(self, input_embeddings):
        output_transform = self.lin(input_embeddings)
        output = self.head(output_transform)
        return output


class ClassificationModel(nn.Module):
    def __init__(self, embedding_model, classification_model, debiasing_model=None, do_debiasing=False,
                 fine_tuning=False):
        super(ClassificationModel, self).__init__()
        self.embedding_model = embedding_model
        self.classification_model = classification_model
        self.debiasing_model = debiasing_model
        self.do_debiasing = do_debiasing
        self.fine_tuning = fine_tuning

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            embedding = self.embedding_model(input_ids, attention_mask, token_type_ids)[1]
        if self.do_debiasing:
            embedding = self.debiasing_model.debias(embedding)
        output = self.classification_model(embedding)
        return output
