import os
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
from attacks import pgd, fgsm, mi_fgsm
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
import seaborn as sns
from src.helper_functions.nuswide_asl import NusWideFiltered
from create_q2l_model import create_q2l_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)

parser = argparse.ArgumentParser()
########################## ARGUMENTS #############################################

# MSCOCO 2014
parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
parser.add_argument('--model_path', type=str, default='./models/tresnetl-asl-mscoco-epoch80')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--num-classes', default=80)
parser.add_argument('--dataset_type', type=str, default='MSCOCO_2014')
parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# PASCAL VOC2007
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='../VOC2007')
# parser.add_argument('--model-path', default='./models/tresnetxl-asl-voc-epoch80', type=str)
# parser.add_argument('--model_name', type=str, default='tresnet_xl')
# parser.add_argument('--num-classes', default=20)
# parser.add_argument('--dataset_type', type=str, default='PASCAL_VOC2007')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# NUS_WIDE
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='../NUS_WIDE')
# parser.add_argument('--model_path', type=str, default='./models/tresnetl-asl-nuswide-epoch80')
# parser.add_argument('--model_name', type=str, default='tresnet_l')
# parser.add_argument('--num-classes', default=81)
# parser.add_argument('--dataset_type', type=str, default='NUS_WIDE')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')


# IMPORTANT PARAMETERS!
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
args = parse_args(parser)

########################## SETUP THE MODELS AND LOAD THE DATA #####################

print('Model = ASL')
state = torch.load(args.model_path, map_location='cpu')
asl = create_model(args).cuda()
model_state = torch.load(args.model_path, map_location='cpu')
asl.load_state_dict(model_state["state_dict"])
asl.eval()
args.model_type = 'asl'
model = asl

# print('Model = Q2L')
# q2l = create_q2l_model('config_coco.json')
# args.model_type = 'q2l'
# model = q2l

################ DATASET LOADING ############################

if args.dataset_type == 'MSCOCO_2014':

    instances_path = os.path.join(args.data, 'annotations/instances_train2014.json')
    data_path = '{0}/train2014'.format(args.data)

    dataset = CocoDetectionFiltered(data_path,
                                instances_path,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
elif args.dataset_type == 'PASCAL_VOC2007':

    dataset = Voc2007Classification('trainval',
                                    transform=transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor(),
                ]), train=True)

elif args.dataset_type == 'NUS_WIDE':
    
    dataset = NusWideFiltered('train', transform=transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor()])
    )


# Pytorch Data loader
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

################ EXPERIMENT VARIABLES  ########################

NUMBER_OF_SAMPLES = 1024

#############################  EXPERIMENT LOOP #############################

confidences = torch.zeros((NUMBER_OF_SAMPLES, args.batch_size, 80)) 
sample_count = 0

for i, (tensor_batch, labels) in enumerate(data_loader):
    tensor_batch = tensor_batch.to(device)
    target = torch.zeros((args.batch_size, args.num_classes))
    target[:, 0:17] = 1

    if sample_count >= NUMBER_OF_SAMPLES:
        break

    # confidences_b4 = torch.sigmoid(model(tensor_batch))
    adversarials = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(weight=None), eps=0.005, device='cuda')

    with torch.no_grad():            
        confidences[i] = (torch.sigmoid(model(adversarials)) > 0.5).int()

    sample_count += args.batch_size
    print(i)

print(torch.sum(confidences, dim=(0,1)))


# batch_size = 16
# tensor([995., 774., 976., 432., 201., 805., 543., 861., 460., 781., 572., 647.,
#         534., 809., 584., 410., 548.,  18.,   7.,   7.,   9.,   1.,  18.,  18.,
#           4.,  16.,   3.,   7.,   6.,   9.,   1.,   2.,   9.,   5.,   4.,   8.,
#          14.,   4.,  15.,   4.,   5.,   8.,   4.,   0.,   0.,   1.,   9.,   0.,
#           5.,   2.,   4.,   3.,   3.,  16.,  15.,   7.,   2.,   5.,  11.,   7.,
#           6.,   9.,   6.,   9.,   2.,   2.,   7.,   3.,   1.,   0.,   0.,   2.,
#           0.,   3.,  15.,   2.,   2.,   6.,   0.,   0.])


# batch_size = 1
# tensor([995., 771., 979., 432., 190., 808., 539., 857., 445., 786., 565., 643.,
#         526., 807., 565., 405., 544.,  18.,   7.,   7.,  10.,   1.,  18.,  17.,
#           3.,  16.,   4.,   6.,   6.,   9.,   1.,   2.,   8.,   4.,   4.,   8.,
#          14.,   4.,  15.,   2.,   5.,   8.,   4.,   0.,   0.,   1.,   9.,   0.,
#           5.,   2.,   4.,   2.,   3.,  16.,  15.,   7.,   2.,   6.,  11.,   7.,
#           6.,   9.,   6.,   9.,   2.,   2.,   7.,   3.,   1.,   0.,   0.,   2.,
#           0.,   3.,  13.,   2.,   2.,   5.,   0.,   0.])
