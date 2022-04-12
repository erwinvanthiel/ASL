import os
import sys
sys.path.append('../')
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from attacks import pgd, fgsm, mi_fgsm
from mlc_attack_losses import SigmoidLoss, HybridLoss, HingeLoss, LinearLoss, MSELoss, SmartLoss
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
from src.helper_functions.voc import Voc2007Classification
from create_q2l_model import create_q2l_model
from src.helper_functions.nuswide_asl import NusWideFiltered

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser()

# MSCOCO 2014
parser.add_argument('--model_path', type=str, default='./models/tresnetl-asl-mscoco-epoch80')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--num-classes', default=80)
parser.add_argument('--dataset_type', type=str, default='MSCOCO_2014')
parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# PASCAL VOC2007
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='../VOC2007')
# parser.add_argument('attack_type', type=str, default='PGD')
# parser.add_argument('--model-path', default='./models/tresnetxl-asl-voc-epoch80', type=str)
# parser.add_argument('--model_name', type=str, default='tresnet_xl')
# parser.add_argument('--num-classes', default=20)
# parser.add_argument('--dataset_type', type=str, default='PASCAL_VOC2007')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# # NUS_WIDE
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='../NUS_WIDE')
# parser.add_argument('attack_type', type=str, default='pgd')
# parser.add_argument('--model_path', type=str, default='./models/tresnetl-asl-nuswide-epoch80')
# parser.add_argument('--model_name', type=str, default='tresnet_l')
# parser.add_argument('--num-classes', default=81)
# parser.add_argument('--dataset_type', type=str, default='NUS_WIDE')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')


# IMPORTANT PARAMETERS!
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=5, type=int,
					metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
					help='number of data loading workers (default: 16)')
args = parse_args(parser)

########################## SETUP THE MODELS AND LOAD THE DATA #####################

# print('Model = ASL')
# state = torch.load(args.model_path, map_location='cpu')
# asl = create_model(args).cuda()
# model_state = torch.load(args.model_path, map_location='cpu')
# asl.load_state_dict(model_state["state_dict"])
# asl.eval()
# args.model_type = 'asl'
# model = asl

print('Model = Q2L')
q2l = create_q2l_model('../config_coco.json')
args.model_type = 'q2l'
model = q2l

ml_cw_flips = []
mi_fgsm_flips = []

for i in range(5):
	print(i)
	clean = torch.tensor(np.load("adv/q2l/MSCOCO_2014/ml_cw_lambda_fixedclean{0}.npy".format(i)))
	adv = torch.tensor(np.load("adv/q2l/MSCOCO_2014/ml_cw_lambda_fixedadv{0}.npy".format(i)))
	pred_clean = (torch.sigmoid(model(clean.cuda())) > 0.5).int()

	# get results cw attack
	pred_cw = (torch.sigmoid(model(adv.cuda())) > 0.5).int()

	# attack with mifgsm and get results
	epsilon = torch.max(torch.abs(clean - adv))
	target = 1 - pred_clean
	mi_fgsm_adv = mi_fgsm(model, clean, target, eps=epsilon)
	mi_fgsm_adv1 = mi_fgsm(model, clean, target, loss_function=LinearLoss(), eps=epsilon)
	pred_mi_fgsm = (torch.sigmoid(model(mi_fgsm_adv.cuda())) > 0.5).int()
	pred_mi_fgsm1 = (torch.sigmoid(model(mi_fgsm_adv1.cuda())) > 0.5).int()

	# store flips
	flips_cw = torch.sum(torch.logical_xor(pred_clean,pred_cw)).item()
	flips_mifgsm = torch.sum(torch.logical_xor(pred_clean,pred_mi_fgsm)).item()
	flips_mifgsm1 = torch.sum(torch.logical_xor(pred_clean,pred_mi_fgsm1)).item()
	print(flips_cw, flips_mifgsm, flips_mifgsm1)
	ml_cw_flips.append(flips_cw)
	mi_fgsm_flips.append(flips_mifgsm)

print(np.mean(ml_cw_flips))
print(np.mean(mi_fgsm_flips))