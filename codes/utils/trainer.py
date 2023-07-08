import torch
import torch.nn.functional as F
import time
import sys


class Trainer():
    def __init__(self, model, flipper, optimizer, epochs, device="cpu", use_flip=False):
        self.model = model.to(device)
        self.filpper = flipper
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.use_flip = use_flip
        
    def train(self, train_data, valid_data):
        for epoch in range(self.epochs):
            self.model.train()
            num_true, num_total = 0, 0
            loss_sum = 0.
            start = time.time()
            
            cnt = 0
            total = len(train_data)
            for fig_data, tokens, seq, mask, labels in train_data:
                print("\repoch %d: %d%%" % (epoch+1, int(cnt / total * 100)), end="")
                for i in range(int(cnt / total * 100)):
                    print("▋", end="")
                sys.stdout.flush()
                
                txt_data = {
                    "input_ids": tokens.to(self.device),
                    "token_type_ids": seq.to(self.device),
                    "attention_mask": mask.to(self.device)
                }
                fig_data, txt_data, labels = fig_data.to(self.device), txt_data, labels.to(self.device)
                out = self.model(fig_data, txt_data)
                num_true += torch.sum(torch.eq(torch.argmax(out, dim=1), labels).int())
                num_total += fig_data.size(0)
                
                loss = F.cross_entropy(out, labels)
                loss_sum += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.use_flip:
                    # flip
                    fig_data = self.filpper(fig_data)
                    out = self.model(fig_data, txt_data)
                    num_true += torch.sum(torch.eq(torch.argmax(out, dim=1), labels).int())
                    num_total += fig_data.size(0)
                    
                    loss = F.cross_entropy(out, labels)
                    loss_sum += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                del fig_data, tokens, seq, mask, labels, txt_data, out, loss
                cnt += 1
            print()
                
            if (epoch + 1) % (self.epochs // 10) == 0:
                time_cost = (time.time() - start) / 60
                train_acc = num_true / num_total
                valid_acc = self.valid(valid_data)
                print("Epoch %d: loss %f train acc %f%% valid acc %f%% time cost %f min" % (epoch+1, loss_sum, train_acc*100, valid_acc*100, time_cost))
                
                
    def valid(self, valid_data, mode=0):
        self.model.eval()
        num_true, num_total = 0, 0
        
        for fig_data, tokens, seq, mask, labels in valid_data:
            txt_data = {
                "input_ids": tokens.to(self.device),
                "token_type_ids": seq.to(self.device),
                "attention_mask": mask.to(self.device)
            }
            fig_data, txt_data, labels = fig_data.to(self.device), txt_data, labels.to(self.device)
            out = torch.argmax(self.model(fig_data, txt_data, mode), dim=1)
            num_true += torch.sum(torch.eq(out, labels).int())
            num_total += fig_data.size(0)
            
            del fig_data, tokens, seq, mask, labels, txt_data, out
        
        return num_true / num_total
    
    def test(self, test_data):
        self.model.eval()
        output = []
        
        for fig_data, tokens, seq, mask in test_data:
            txt_data = {
                "input_ids": tokens.to(self.device),
                "token_type_ids": seq.to(self.device),
                "attention_mask": mask.to(self.device)
            }
            fig_data, txt_data = fig_data.to(self.device), txt_data
            out = torch.argmax(self.model(fig_data, txt_data), dim=1)
            output.append(out.to("cpu"))
            del fig_data, tokens, seq, mask, txt_data, out
        
        return torch.cat(output, dim=0)
            
