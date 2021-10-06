import numpy as np
import torch

#======================== MixUp ===================================
def onehot(targets, num_classes):
    assert isinstance(targets, torch.LongTensor)
    return torch.zeros(targets.size()[0], num_classes).scatter_(1, targets.view(-1, 1), 1)

def mixup (inputs, targets, num_classes, alpha = 2):
    s = inputs.size()[0] #create matrix weight

    weight = torch.Tensor(np.random.beta(alpha, alpha, s))
    index = np.random.permutation(s)

    x1, x2 = inputs, inputs[index, :, :, :]
    y1, y2 = onehot(targets, num_classes), onehot(targets[index,], num_classes)

    weight = weight.view(s, 1, 1, 1)
    inputs = weight*x1 + (1-weight) * x2
    
    weight = weight.view(s,1)
    targets = weight * y1 + (1-weight) *y2

    return inputs, targets

#======================== CutMix ====================================
def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] #  width
    H = size[3] # height
    cut_rat = np.sqrt(1. - lam)  # 
    cut_w = np.int(W * cut_rat)  # 
    cut_h = np.int(H * cut_rat)  # 

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def CutMix(inputs, targets, beta):
    """ Generate a CutMix augmented image from a batch
    Args:
        - image_batch: a batch of input images
        - image_batch_labels: labels corresponding to the image batch
        - beta: a parameter of Beta distribution.
    Returns:
        - CutMix image batch, updated labels
    """
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = np.random.permutation(len(image_batch))
    target_a = targets
    target_b = targets[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs[0].shape, lam)
    inputs_update = inputs.copy()
    inputs_update[:, bbx1:bbx2, bby1:bby2, :] = inputs[rand_index, bbx1:bbx2, bby1:bby2, :]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.shape[1] * inputs.shape[2]))
    label = target_a * lam + target_b * (1. - lam)

    return inputs_update, label

# ===================================================================
def mosaic(inputs, targets, num_classes,):


