import torch
import os, time
import numpy as np
from tqdm import tqdm
#from Network import MyNetwork
from Network import  MyNetwork
from ImageUtils import parse_record
import torch.nn as nn
import torch.optim as optim
from ImageUtils import parse_record
import torch.nn.functional as F

""" This script defines the training, validation and testing process.
"""

class MyModel(nn.Module):
    def __init__(self, configs):
	    
        super(MyModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs = configs
        self.network = MyNetwork().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.lr,nesterov=True,weight_decay=self.weight_decay,momentum = 0.9)
        self.criterion = nn.CrossEntropyLoss()
        ### YOUR CODE HERE
    def model_setup(self,configs):
        self.lr = configs['learning_rate']
        self.weight_decay = configs['weight_decay']
        self.batch_size = configs['batch_size']
        self.epoch = configs['epoch']
        self.save_interval = configs['save_interval']
        self.save_dir = configs['save_dir']

    
    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        print(self.device)
        self.network.train()
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.batch_size

        print('### Training... ###')
        for epoch in range(1, self.epoch + 1):
            epoch_loss = 0
            start_time = time.time()
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            #l2_slope = 0

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, curr_x_train.shape[0])
                batch_data = curr_x_train[start_idx: end_idx]
                batch_labels = torch.tensor(curr_y_train[start_idx: end_idx], dtype=torch.int64).to(self.device)
                parsed_batch_data = torch.tensor(np.array([parse_record(record, True) for record in batch_data]), dtype=torch.float32).to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.network(parsed_batch_data,training = True)
                batch_loss = self.criterion(y_pred, batch_labels)
                #l2_lambda = self.weight_decay
                #l2_slope = torch.tensor(0.).to('cuda')
                #for param in self.network.parameters():
                    #l2_slope += torch.norm(param) 
                #l2_loss = l2_lambda * l2_slope

                batch_loss = batch_loss #+ l2_loss
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss += batch_loss.item()
                print('Epoch {:d}, Batch {:d}/{:d}, Loss: {:.6f}'.format(epoch, i+1, num_batches, batch_loss.item()))

            duration = time.time() - start_time
            avg_epoch_loss = epoch_loss / num_batches
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, avg_epoch_loss, duration))

            # Evaluate on validation data
            #if x_valid is not None and y_valid is not None:
                #self.evaluate(x_valid, y_valid, [epoch])

            # Save the model if it has the best accuracy on validation data
            if epoch%10==0:
                self.save(epoch)
            if epoch%80==0 or epoch%180==0 or epoch%270==0:
                for param_group in self.optimizer.param_groups:
                    self.lr = self.lr/10
                    param_group['lr'] = self.lr
            




    def evaluate(self, x, y, checkpoint_num_list):
        #self.network.eval()
        #print(x.shape)
        print('Evaluation')
        #self.network = MyNetwork().to(self.device)
        #print(self.network)
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.save_dir, 'model-%d.ckpt'%(checkpoint_num))
            #checkpointfile = os.path.join(self.save_dir, 'model_92_sgd.pkl')
            self.load(checkpointfile)
            self.network.load_state_dict(torch.load(checkpointfile))
            self.network.eval()
            #print(list(self.network.parameters()))
            num_batches = x.shape[0] // self.batch_size
            with torch.no_grad():
                preds = []
                correct = 0
                total = 0
                for i in range(num_batches):
                    start_idx = i * self.batch_size
                    end_idx = min((i + 1) * self.batch_size, x.shape[0])
                    batch_data = x[start_idx: end_idx]
                    batch_labels = torch.tensor(y[start_idx: end_idx], dtype=torch.int64).to(self.device)
                    #print(batch_data.shape)
                    #print(batch_labels.shape)
                    parsed_batch_data = torch.tensor(np.array([parse_record(record, False) for record in batch_data]), dtype=torch.float32).to(self.device)
                    #print(parsed_batch_data.shape)
                    output = self.network(parsed_batch_data)
                    values,predict = torch.max(output.data,1)
                    #print(predict)
                    total = total + batch_data.shape[0]
                    #print(batch_labels)
                    #print(predict)
                    correct += (predict == batch_labels.data).sum().item()
                    #print(correct)
                    #preds.append(predict)
                    #for pred in predict:
                        #preds.append(pred.item())



            #for i in tqdm(range(x.shape[0])):
                #output = self.predict_prob(x[i])

                #values,predict = torch.max(output.data,1)
                #preds.append(predict.item())

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            accuracy = correct / total
            #accuracy = torch.sum(preds==y)/x.shape[0]
            
            print('Test accuracy: {:.4f}'.format(accuracy))
            #return accuracy
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.save_dir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

    def predict_prob(self, x,checkpoint_num):
        checkpointfile = os.path.join(self.save_dir, 'model-%d.ckpt'%(checkpoint_num))
            #checkpointfile = os.path.join(self.save_dir, 'model_92_sgd.pkl')
        self.load(checkpointfile)
        self.network.load_state_dict(torch.load(checkpointfile))
        self.network.eval()
        output = []
        with torch.no_grad():
            parsed_data = torch.tensor(np.array([parse_record(record, False) for record in x]), dtype=torch.float32).to(self.device)
            for i in range(x.shape[0]):
                logits = self.network(parsed_data[i].reshape(1,3,32,32))
                probs = F.softmax(logits, dim=1)
                output.append(probs.cpu().numpy())

            
            print(output)
            output = np.array(output)
            return output
