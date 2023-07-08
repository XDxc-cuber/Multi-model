import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, hidden_dim=512, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data: dict):
        bert_out = self.bert(**data)[1]
        return self.dropout(self.linear(bert_out))
    
    
    
if __name__ == "__main__":
    data = torch.load("data/test_txt.pt")
    for k in data.keys():
        data[k] = torch.tensor(data[k])[:5]
    model = BertClassifier(3)
    print(model(data).shape)
    print(model)
    