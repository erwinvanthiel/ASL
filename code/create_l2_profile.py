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

# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])

################ EXPERIMENT VARIABLES ########################

alpha = float(448 / 2560)
NUMBER_OF_SAMPLES = 100
sample_count = 0
samples = torch.zeros((NUMBER_OF_SAMPLES, 3, args.image_size, args.image_size))
targets = torch.zeros((NUMBER_OF_SAMPLES, args.num_classes))
flipped_labels_patient = [0]
flipped_labels_greedy = [0]
flipped_labels_patient_deltas = []

#############################  EXPERIMENT LOOP #############################


# Fetch 100 samples with their respective attack targets
for i, (tensor_batch, _) in enumerate(data_loader):
    tensor_batch = tensor_batch.to(device)

    if sample_count >= NUMBER_OF_SAMPLES:
        break

    samples[i] = tensor_batch

    # Do the inference
    with torch.no_grad():
        output = torch.sigmoid(model(tensor_batch))
        pred = (output > args.th).int()
        target = torch.clone(pred).detach()
        target = 1 - target
        targets[i] = target.cpu()

    sample_count += 1

iters = 0
interval = 10
converged = False

bce_samples = torch.clone(samples)
linear_samples = torch.clone(samples)
bce_grads = torch.zeros((100, 3, args.image_size, args.image_size)).to(device)
linear_grads = torch.zeros((100, 3, args.image_size, args.image_size)).to(device)

while not converged:
    iters += 1
    # print('iteration', iters)
    bce_flips = 0
    linear_flips = 0

    mu = 1.0
    L1 = torch.nn.BCELoss()
    L2 = LinearLoss()

    for i in range(NUMBER_OF_SAMPLES):

        # MI-FGSM for bce
        image = bce_samples[i].unsqueeze(0).to(device)
        image.requires_grad = True
        outputs = torch.sigmoid(model(image)).to(device)

        model.zero_grad()
        cost1 = L1(outputs, targets[i].unsqueeze(0).float().to(device).detach())
        cost1.backward()

        # normalize the gradient
        new_g = image.grad / torch.sqrt(torch.sum(image.grad ** 2))

        # update the gradient
        bce_grads[i] = mu * bce_grads[i] + new_g

        # perform the step, and detach because otherwise gradients get messed up.
        image = (image - alpha * bce_grads[i]).detach()

        bce_samples[i] = image


        # MI-FGSM for linear
        image = linear_samples[i].unsqueeze(0).to(device)
        image.requires_grad = True
        outputs = torch.sigmoid(model(image)).to(device)

        model.zero_grad()
        cost2 = L2(outputs, targets[i].unsqueeze(0).float().to(device).detach())
        cost2.backward()

        # normalize the gradient
        new_g = image.grad / torch.sqrt(torch.sum(image.grad ** 2))

        # update the gradient
        linear_grads[i] = mu * linear_grads[i] + new_g

        # perform the step, and detach because otherwise gradients get messed up.
        image = (image - alpha * linear_grads[i]).detach()

        linear_samples[i] = image

        if iters % interval == 0:

            with torch.no_grad():
                adv = torch.clamp(bce_samples[i], min=0, max=1)
                bce_output = torch.sigmoid(model(adv.to(device).unsqueeze(0)))
                bce_pred = (bce_output > args.th).int()
                bce_flips += torch.sum(torch.logical_xor(bce_pred.cpu(), 1 - targets[i])).item()

                adv = torch.clamp(linear_samples[i], min=0, max=1)
                linear_output = torch.sigmoid(model(adv.to(device).unsqueeze(0)))
                linear_pred = (linear_output > args.th).int()
                linear_flips += torch.sum(torch.logical_xor(linear_pred.cpu(), 1 - targets[i])).item()

    if iters % interval == 0:
        flipped_labels_patient.append(bce_flips / NUMBER_OF_SAMPLES)
        flipped_labels_greedy.append(linear_flips / NUMBER_OF_SAMPLES)
        print(flipped_labels_patient)


    if len(flipped_labels_patient) >= 2:
        size = len(flipped_labels_patient)
        delta = flipped_labels_patient[size - 1] - flipped_labels_patient[size - 2]
        flipped_labels_patient_deltas.append(delta)

        if delta / flipped_labels_patient_deltas[0] < 0.01:
            converged = True


EPSILON_VALUES = [i * interval * alpha for i in range(len(flipped_labels_patient))]

coefs = poly.polyfit(EPSILON_VALUES, np.maximum(np.array(flipped_labels_patient),np.array(flipped_labels_patient)), 4)
print(EPSILON_VALUES)

np.save('experiment_results/{0}-{1}-l2-profile-flips'.format(args.model_type, args.dataset_type), np.maximum(np.array(flipped_labels_patient),np.array(flipped_labels_patient)))
np.save('experiment_results/{0}-{1}-l2-profile'.format(args.model_type, args.dataset_type), coefs)
np.save('experiment_results/{0}-{1}-l2-profile-epsilons'.format(args.model_type, args.dataset_type), EPSILON_VALUES)
