import os
import copy
import torch
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torchvision
import torch.nn as nn
from dataset import *
from efficientNet import *
from tqdm import tqdm
import torch.distributed as dist
from torchsummary import summary


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

def train_val(model, params, device):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    path2weights=params["path2weights"]

    # history of loss values
    loss_history = {
            "train": [],
            "val" : []
            }

    acc_history = {
            "train" : [],
            "val" : []
            }

    if os.path.exists(path2weights):
        print("Load saved weight")
        model.load_state_dict(torch.load(path2weights))
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


    #load best model weights
    model.load_state_dict(best_model_wts)

    return model, loss_history, acc_history
def main():

    batch_size = 32

    transform = transforms.Compose([
        transforms.CenterCrop(10),
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

    model = efficientnet_b5()
    model = model.to(device)

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


    model, loss_history, acc_history = train_val(model, params, device)

    # Plot train-val loss
    num_epochs = params["num_epochs"]
    plt.title('Train-Val Loss')
    plt.plot(range(1, num_epochs+1), loss_history['train'], label='train')
    plt.plot(range(1, num_epochs+1), loss_history['val'], label='val')
    plt.ylabel('Loss')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.show()
    
    # plot train-val accuracy
    plt.title('Train-Val Accuracy')
    plt.plot(range(1, num_epochs+1), acc_history['train'], label='train')
    plt.plot(range(1, num_epochs+1), acc_history['val'], label='val')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()





