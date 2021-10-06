import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import glob

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


class STLCustomData(Dataset):
    def __init__(self, root, type_, transform = None):
        self.root = root
        self.data_path = os.path.join(self.root, type_)
        self.list_folder = os.listdir(self.data_path)
        images = []
        labels = []
        for folder in self.list_folder:
            image_path = os.path.join(self.data_path, folder)
            images += glob.glob(f"{image_path}/*")
            labels += [int(folder)-1] * len(os.listdir(os.path.join(self.data_path, folder)))
        self.images = images
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image_file, label = self.images[idx], self.labels[idx]
        image = Image.open(image_file)

        if self.transform is not None:
            image = self.transform(image)
        return image, label

# check sample images
def show(img, y=None):
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg_tr)
    
    if y is not None:
        plt.title('labels:' + str(y))
    

if __name__ =='__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    root = os.path.join('/hdd/image_classification/train_data')
    dataset = STLCustomData(root, "train", transform)
    
    np.random.seed(10)
    torch.manual_seed(0)
    
    train_ds = STLCustomData(root, "train")
    val_ds = STLCustomData(root, "test")

    grid_size=4
    rnd_ind = np.random.randint(0, len(train_ds), grid_size)
    
    x_grid = [train_ds[i][0] for i in rnd_ind]
    y_grid = [val_ds[i][1] for i in rnd_ind]
    
    x_grid = utils.make_grid(x_grid, nrow=grid_size, padding=2)
    plt.figure(figsize=(10,10))
    show(x_grid, y_grid)




