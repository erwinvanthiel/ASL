import os
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from src.models import create_model
from src.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('data', metavar='DIR', help='path to dataset', default='/home/MSCOCO_2014/')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_m')
parser.add_argument('--model-path', default='./mlc-model-epoch100', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')

args = parser.parse_args()
args.do_bottleneck_head = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_INDEX = 60 # label for donut object


# Configure mlc model
model = create_model(args).cuda()
model_state = torch.load("mlc-model-epoch100", map_location=device)
model.load_state_dict(model_state["state_dict"])
model.eval()
model.to(device)

# Generate adversarials with targeted PGD attack
# source: https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb
def create_adversarial_examples(model, images, target, eps=0.3, alpha=2/255, iters=40, device='cpu'):
    images = images.to(device)
    target = target.to(device).float()
    model = model.to(device)
    loss = nn.BCELoss()
        
    ori_images = images.data
        
    for i in range(iters):    
        images.requires_grad = True
        outputs = model(images).to(device)

        model.zero_grad()
        cost = loss(outputs, target).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images


# Load the data
instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
# data_path_train = args.data
data_path_train = f'{args.data}/train2014'  # args.data
train_dataset = CocoDetection(data_path_train,
                              instances_path_train,
                              transforms.Compose([
                                  transforms.Resize((args.image_size, args.image_size)),
                                  CutoutPIL(cutout_factor=0.5),
                                  RandAugment(),
                                  transforms.ToTensor(),
                                  # normalize,
                              ]))
print("len(train_dataset)): ", len(train_dataset))

# Pytorch Data loader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)


# Attack with a single batch
(images, labels) = next(iter(train_loader))
labels = labels.max(dim=1)[0]
print(labels.shape)
print(images.shape)



target_labels = labels
target_labels[:, TARGET_INDEX] = 1

adversarials = create_adversarial_examples(model, images, target_labels, device=device)
print(adversarials.shape)
new_labels = model(adversarials)

print(new_labels.shape)




# print(torch.sum(labels[:, TARGET_INDEX]))
# print(torch.sum(new_labels[:, TARGET_INDEX]))

