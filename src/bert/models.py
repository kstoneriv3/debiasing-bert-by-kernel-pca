from transformers import BertModel
import torch.nn as nn


class EmbeddingModel(nn.Module):
    def __init__(self, model_name, batch_size,device):
        super(EmbeddingModel, self).__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.embedding_model = BertModel.from_pretrained("bert-base-uncased")
        self.embedding_model.to(self.device)
        self.embedding_model.eval()


    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.embedding_model.forward(input_ids=input_ids.reshape(self.batch_size, -1).to(self.device),
                                            attention_mask=attention_mask.reshape(self.batch_size, -1).to(self.device),
                                            token_type_ids=token_type_ids.reshape(self.batch_size, -1).to(self.device))
