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


