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
mpl.style.use('classic')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

torch.manual_seed(11)
torch.cuda.manual_seed_all(11)
np.random.seed(11)

########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser()

parser.add_argument('classifier', type=str, default='asl_coco')
parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
parser.add_argument('--dataset_type', type=str, default='MSCOCO_2014')



# IMPORTANT PARAMETERS!
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
args = parse_args(parser)


########################## SETUP THE MODELS  #####################


if args.classifier == 'asl_coco'

    asl, config = create_asl_model('asl_coco.json')
    asl.eval()
    args.model_type = 'asl'
    model = asl

elif args.classifier == 'asl_nuswide':
    asl, config = create_asl_model('asl_nuswide.json')
    asl.eval()
    args.model_type = 'asl'
    model = asl

elif args.classifier == 'asl_voc':
    asl, config = create_asl_model('asl_voc.json')
    asl.eval()
    args.model_type = 'asl'
    model = asl

elif args.classifier == 'q2l_coco':
    q2l = create_q2l_model('q2l_coco.json')
    args.model_type = 'q2l'
    model = q2l

elif args.classifier == 'q2l_nuswide':
    q2l = create_q2l_model('q2l_nuswide.json')
    args.model_type = 'q2l'
    model = q2l

args_dict = {**vars(args), **vars(config)}
args = types.SimpleNamespace(**args_dict)


########################## LOAD THE DATASET  #####################

if args.dataset_type == 'MSCOCO_2014':

    instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
    data_path = '{0}/val2014'.format(args.data)

    dataset = CocoDetectionFiltered(data_path,
                                instances_path,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))

elif args.dataset_type == 'VOC2007':

    dataset = Voc2007Classification('trainval',
                                    transform=transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor(),
                ]), train=True)

elif args.dataset_type == 'NUS_WIDE':
    
    dataset = NusWideFiltered('val', path=args.data, transform=transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor()])
    )

# Pytorch Data loader
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(193579088 / 53750896)


# LOAD THE DATASET WITH DESIRED FILTER

if args.dataset_type == 'MSCOCO_2014':

    instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
    data_path = '{0}/val2014'.format(args.data)

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
    
    dataset = NusWideFiltered('val', transform=transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor()])
    )

# Pytorch Data loader
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)


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
