import utils.label_map as label_map
import torch
from PIL import Image
import torchvision.transforms as transforms
import langid
from google_trans_new import google_translator
from transformers import BertTokenizer
import random
import torch.utils.data as Data



translator = google_translator()


def load_data(path: str, batch_size: int=8, random_seed: int=666, valid_rate: float=0.1):
    train_txt = torch.load(path + "/train_txt.pt")
    test_txt = torch.load(path + "/test_txt.pt")
    train_fig = torch.load(path + "/train_fig.pt")
    test_fig = torch.load(path + "/test_fig.pt")
    train_label = torch.load(path + "/train_label.pt")
    
    num_train, num_test = train_fig.size(0), test_fig.size(0)
    
    # shuffle
    for k in train_txt.keys():
        random.seed(random_seed)
        random.shuffle(train_txt[k])
    random.seed(random_seed)
    random.shuffle(train_fig)
    random.seed(random_seed)
    random.shuffle(train_label)
    
    # split
    valid_num = int(valid_rate * num_train)
    valid_fig, train_fig = train_fig[:valid_num], train_fig[valid_num:]
    valid_label, train_label = train_label[:valid_num], train_label[valid_num:]
    valid_tokens, train_tokens = torch.tensor(train_txt["input_ids"][:valid_num]).int(), torch.tensor(train_txt["input_ids"][valid_num:]).int()
    valid_seq, train_seq = torch.tensor(train_txt["token_type_ids"][:valid_num]).int(), torch.tensor(train_txt["token_type_ids"][valid_num:]).int()
    valid_mask, train_mask = torch.tensor(train_txt["attention_mask"][:valid_num]).int(), torch.tensor(train_txt["attention_mask"][valid_num:]).int()
    
    # get dataloader
    train_data = Data.DataLoader(Data.TensorDataset(train_fig, train_tokens, train_seq, train_mask, train_label), batch_size=batch_size, shuffle=False)
    valid_data = Data.DataLoader(Data.TensorDataset(valid_fig, valid_tokens, valid_seq, valid_mask, valid_label), batch_size=batch_size, shuffle=False)
    test_data = Data.DataLoader(Data.TensorDataset(test_fig, torch.tensor(test_txt["input_ids"]).int(), torch.tensor(test_txt["token_type_ids"]).int(), torch.tensor(test_txt["attention_mask"]).int()), batch_size=batch_size, shuffle=False)
    
    return train_data, valid_data, test_data
    

def translate_seq(seq: str):
    seq = " ".join([w.strip("#") for w in seq.split(" ")])
    lang, _ = langid.classify(seq)
    if lang != "en":
        print(lang, seq)
        seq = translator.translate(seq, lang_tgt="en")
        
    return seq

def save_processed_data(root: str):
    train_id, test_id, train_label = get_id_label(root)
    train_label = torch.tensor(train_label).long()
    torch.save(train_label, root + "/train_label.pt")
    
    fig_trsf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    fig_train, txt_train, fig_test, txt_test = [], [], [], []
    for uid in train_id:
        fig_data = Image.open(root + ("/data/%d.jpg" % uid))
        fig_train.append(fig_trsf(fig_data))
        with open(root + ("/data/%d.txt" % uid), encoding="gb18030") as f:
            txt_train.append(translate_seq(f.readlines()[0].strip()))
    for uid in test_id:
        fig_data = Image.open(root + ("/data/%d.jpg" % uid))
        fig_test.append(fig_trsf(fig_data))
        with open(root + ("/data/%d.txt" % uid), encoding="gb18030") as f:
            txt_test.append(translate_seq(f.readlines()[0].strip()))
    
    
    # 文本转tokens
    txt_train = tokenizer(txt_train, padding=True, max_length=130)
    txt_test = tokenizer(txt_test, padding=True, max_length=130)
    
    # 保存与处理文件
    torch.save(torch.stack(fig_train, dim=0), root + "/train_fig.pt")
    torch.save(torch.stack(fig_test, dim=0), root + "/test_fig.pt")
    torch.save(txt_train, root + "/train_txt.pt")
    torch.save(txt_test, root + "/test_txt.pt")

def get_id_label(root: str):
    train_id, test_id, train_label = [], [], []
    
    with open(root + "/train.txt") as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            train_id.append(int(line.split(",")[0]))
            train_label.append(label_map.label_map[line.split(",")[1]])
    
    with open(root + "/test_without_label.txt") as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            test_id.append(int(line.split(",")[0]))
            
    return train_id, test_id, train_label
    
def show_data_distributions(root: str):
    train_id, test_id, train_label = [], [], []
    
    with open(root + "/train.txt") as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            train_id.append(int(line.split(",")[0]))
            train_label.append(label_map.label_map[line.split(",")[1]])
    
    with open(root + "/test_without_label.txt") as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            test_id.append(int(line.split(",")[0]))
    
    print("Train data: %d" % len(train_id))
    print("Test data: %d" % len(test_id))
    print("Positive num: %d" % train_label.count(0))
    print("Neutral num: %d" % train_label.count(1))
    print("Negative num: %d" % train_label.count(2))
            

if __name__ == '__main__':
    # save_processed_data("data")
    show_data_distributions("data")

