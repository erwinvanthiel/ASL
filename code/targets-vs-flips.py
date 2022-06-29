import sys
import _init_paths
import os
import torch
from asl.src.helper_functions.helper_functions import parse_args
from asl.src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from asl.src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from attacks import pgd, fgsm, mi_fgsm, get_top_n_weights, get_weights_from_correlations, l2_mi_fgm
from mlc_attack_losses import SigmoidLoss, HybridLoss, HingeLoss, LinearLoss, MSELoss, SLAM
from sklearn.metrics import auc
from asl.src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
from asl.src.helper_functions.voc import Voc2007Classification
from create_q2l_model import create_q2l_model
from create_asl_model import create_asl_model
from asl.src.helper_functions.nuswide_asl import NusWideFiltered
import numpy.polynomial.polynomial as poly
import numpy.ma as ma
import matplotlib as mpl
import numpy.polynomial.polynomial as poly
import types
from model_and_dataset_loader import parse_model_and_args, load_dataset
mpl.style.use('classic')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

torch.manual_seed(11)
torch.cuda.manual_seed_all(11)
np.random.seed(11)

########################## LOAD DATASET AND MODEL #############################################

args, model = parse_model_and_args()
data_loader = load_dataset(args)



################ EXPERIMENT VARIABLES ########################

NUMBER_OF_SAMPLES = 100
epsilons = np.load('experiment_results/{0}-{1}-l2-profile-epsilons.npy'.format(args.model_type, args.dataset_type))
min_eps = 0.1 * epsilons[len(epsilons) - 1]
EPSILON_VALUES = [min_eps, 3*min_eps, 6*min_eps]
amount_of_targets = [5,10,15,20,25,30,35,40,50,60,70,80]
print(EPSILON_VALUES)
flipped_labels = np.zeros((2, len(EPSILON_VALUES), len(amount_of_targets), NUMBER_OF_SAMPLES))

#############################  EXPERIMENT LOOP #############################

sample_count = 0

# DATASET LOOP
for i, (tensor_batch, labels) in enumerate(data_loader):
    tensor_batch = tensor_batch.to(device)

    if sample_count >= NUMBER_OF_SAMPLES:
        break

    # Do the inference
    with torch.no_grad():
        outputs = torch.sigmoid(model(tensor_batch))
        pred = (outputs > args.th).int()
        target = torch.clone(pred).detach()
        target = 1 - target

    for epsilon_index, epsilon in enumerate(EPSILON_VALUES):

        # process a batch and add the flipped labels for every number of targets
        for amount_id, number_of_targets in enumerate(amount_of_targets):
            weights = get_top_n_weights(outputs, number_of_targets, random=False).to(device)
            adversarials = l2_mi_fgm(model, tensor_batch, target, loss_function=torch.nn.BCELoss(weight=weights), eps=EPSILON_VALUES[epsilon_index], device="cuda").detach()
            # adversarials_r = mi_fgsm(model, tensor_batch, target, loss_function=torch.nn.BCELoss(weight=get_top_n_weights(outputs, number_of_targets, target, random=True).to(device)), eps=EPSILON_VALUES[epsilon_index], device="cuda")
        
            with torch.no_grad():
                # Another inference after the attack
                pred_after_attack = (torch.sigmoid(model(adversarials)) > args.th).int()
                # pred_after_attack_r = (torch.sigmoid(model(adversarials_r)) > args.th).int()
                
                flipped_labels[0, epsilon_index, amount_id, i*args.batch_size:(i+1)*args.batch_size] = torch.sum(torch.logical_xor(pred, pred_after_attack), dim=1).cpu().numpy()
                # flipped_labels[1, epsilon_index, amount_id, i*args.batch_size:(i+1)*args.batch_size] = torch.sum(torch.logical_xor(pred, pred_after_attack_r), dim=1).cpu().numpy()
            
    sample_count += args.batch_size
    print('batch number:',i)

print(flipped_labels)
np.save('experiment_results/l2-targets-vs-flips-{0}-{1}.npy'.format(args.model_type, args.dataset_type),flipped_labels)


