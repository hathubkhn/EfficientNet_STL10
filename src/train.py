from tqdm import tqdm
import copy
from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset import *

import torchvision
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

num_show_img = 5

class_names = {
    "0": "airplane",      
    "1": "bird",   
    "2": "car",   
    "3": "cat",  
    "4": "deer",   
    "5": "dog",   
    "6": "horse",
    "7": "monkey",
    "8": "ship", 
    "9": "truck"
}
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def get_acc(output, target):
    pred = torch.argmax(output, 1)
    acc = (pred == target).sum().item()
    return acc


# should use pytorch_lightning for parallel
def loss_batch(loss_func, output, target, opt = None):
    loss = loss_func(output, target)

    metric_b = get_acc(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, device, opt = None):
    running_loss = 0.0
    running_metric = 0.0

    len_data = len(dataset_dl.dataset)

    for i, (img, label) in enumerate(tqdm(dataset_dl)):
        img = img.to(device)
        label = label.to(device)

        output = model(img)
        loss, metric_b = loss_batch(loss_func, output, label, opt)
        #update running loss
        running_loss += loss

        #update acc
        if metric_b is not None:
            running_metric += metric_b


    loss_epoch = running_loss/float(len_data)
    metric = running_metric/float(len_data)
    
    return loss_epoch, metric

def train_val(model, params, device, patience):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    path2weights=params["path2weights"]

    early_stopping = EarlyStopping(patience = patience, verbose = True, path = path2weights)
    
    # history of loss values
    loss_history = {
            "train": [],
            "val" : []
            }

    acc_history = {
            "train" : [],
            "val" : []
            }

    #a deep copy of weights for the bes performing
    best_model_wts = copy.deepcopy(model.state_dict())

    #initialize best loss to a large value
    best_loss = float('inf')

    #main loop
    for epoch in range(num_epochs):
        # train model
        print("Start training with {}".format(epoch))
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, device, opt)

        #collect loss 
        loss_history["train"].append(train_loss)
        acc_history["train"].append(train_metric)

        #eval model
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, device)
            


        # Store best weight
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            # Store weight into a local file
            torch.save(model.state_dict(), path2weights)
        print("train loss: %.6f, dev loss: %.6f, accuracy_train: %.2f, accuracy_val: %.2f" %(train_loss, val_loss, train_metric, val_metric))
        
        #collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        acc_history["val"].append(val_metric)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break


    #load and save best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_history, acc_history

def main():
    batch_size = 32

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    root = os.path.join(os.path.abspath('../') , 'train_data')
    train_dataset = STLCustomData(root, "train", transform)
    val_dataset = STLCustomData(root, "test", transform)

    # Creating PT data samplers and loaders:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle = True, drop_last = True)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                    shuffle = True, drop_last = True)

    # Define optimizer, loss and model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_name = 'efficientnet-b5'
    image_size = EfficientNet.get_image_size(model_name)

    print(image_size)

    model = EfficientNet.from_pretrained(model_name, num_classes = 10)

    model = nn.DataParallel(model)
    criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')
    optimizer  = torch.optim.Adam(model.parameters(), lr = 1e-3)

    params = {
        "num_epochs" : 150,
        "optimizer": optimizer,
        "loss_func" : criterion,
        "train_dl": train_loader,
        "val_dl": validation_loader,
        "path2weights": "../models/weights.pt"
    }

    patience = 10
    model, loss_history, acc_history = train_val(model, params, device, patience)

if __name__ == '__main__':
    main()



