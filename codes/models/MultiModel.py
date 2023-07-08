import torch
import torch.nn as nn
from models import MobileNetV2, BertClassifier


class MultiModel(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=512, dropout=0.5, device="cpu"):
        super(MultiModel, self).__init__()
        self.fig_model = MobileNetV2.MobileNetV2(hidden_dim, dropout)
        self.txt_model = BertClassifier.BertClassifier(hidden_dim, dropout)
        self.act = nn.ReLU(inplace=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, fig_data, txt_data, mode):
        fig_out = self.fig_model(fig_data)
        txt_out = self.txt_model(txt_data)
        if mode == 1:
            fig_out *= 0
        if mode == 2:
            txt_out *= 0
        concated_out = torch.concat((fig_out, txt_out), dim=1)
        out = self.fc(self.act(concated_out))
        return out
    
    
    
if __name__ == '__main__':
    B = 5
    fig_data = torch.zeros((B, 3, 224, 224))
    txt_data = {
        "input_ids": torch.zeros((B, 130)).int(),
        "token_type_ids": torch.zeros((B, 130)).int(),
        "attention_mask": torch.ones((B, 130)).int(),
    }
    
    model = MultiModel(3, 512)
    out = model(fig_data, txt_data)
    print(out.shape)


