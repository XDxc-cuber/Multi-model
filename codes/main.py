import torch
from torch.optim import Adam
from utils.loadData import load_data
from utils.trainer import Trainer
from models.MultiModel import MultiModel
from torchvision.transforms import RandomHorizontalFlip
import os

from utils.label_map import num_map


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)


data_path = "data"
state_dict_path = "codes/models/state_dict_model.pt"
batch_size = 8
valid_rate = 0.1
dropout = 0.75
random_state = 666
lr = 1e-5
epoch = 10
num_classes = 3
hidden_dim = 512
reg = 1e-3
device = "cuda"

use_flip = True
do_test = False



if __name__ == "__main__":
    setup_seed(random_state)
    print("Loading data...")
    train_data, valid_data, test_data = load_data(data_path, batch_size, random_state, valid_rate)
    model = MultiModel(num_classes, hidden_dim, dropout)
    flipper = RandomHorizontalFlip(1)
    optimizer = Adam(model.parameters(), lr, weight_decay=reg)
    trainer = Trainer(model, flipper, optimizer, epoch, device, use_flip)

    if not os.path.exists(state_dict_path):
        print("Training...")
        trainer.train(train_data, valid_data)
        torch.save(model.state_dict(), state_dict_path)
    else:
        model.load_state_dict(torch.load(state_dict_path))
        
    print("\nValidating...")
    valid_acc = trainer.valid(valid_data, 0)
    print("valid acc %f%%" % (valid_acc * 100))
    valid_acc = trainer.valid(valid_data, 1)
    print("valid acc without fig %f%%" % (valid_acc * 100))
    valid_acc = trainer.valid(valid_data, 2)
    print("valid acc without txt %f%%" % (valid_acc * 100))
    
    if do_test:
        print("Testing...")
        with open(data_path + "/test_without_label.txt") as f:
            lines = f.readlines()
            with open(data_path + "/test_with_label.txt", "w+") as ff:
                ff.write(lines[0])
                out = trainer.test(test_data).to("cpu")
                test_num = out.size(0)
                for i in range(test_num):
                    ff.write(lines[i+1].split(",")[0] + "," + str(num_map[int(out[i])]) + "\n")
                
        
    
    