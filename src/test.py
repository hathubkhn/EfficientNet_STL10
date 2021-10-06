import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch
import torchvision
from PIL import Image
import os
from torchvision import transforms
from torchsummary import summary
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='cat.jpg')
    args = parser.parse_args()
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
    
    init_path = os.path.abspath("../")
    image_file = os.path.join(init_path, "cat.jpg")
    image = Image.open(image_file)
    ori_img = image.copy()
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = transform(image).to(device).unsqueeze(0)
    print(image.shape)
    
    weight_file = os.path.abspath("../models/weights.pt")
    model_name = 'efficientnet-b5'
    model = EfficientNet.from_pretrained(model_name, num_classes = 10)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(weight_file))
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        print(output)
    
    _, preds = torch.max(output, 1)
    
    cv2.putText(ori_img, str(class_names[preds.item()]),(75, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0))
    cv2.imshow('result', ori_img)
    cv2.waitKey(0)
    
    print(class_names[str(preds.item())])
    
if __name__ == '__main__':
    main()
