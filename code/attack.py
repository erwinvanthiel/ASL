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


########################## FUNCTIONS #####################

def plot_confidences(output, output_after):
    output_after = torch.sigmoid(model(adversarials1))
    y1 = output_after.detach().cpu()[0]
    y2 = output.detach().cpu()[0]
    mask1 = ma.where(y1>=y2)
    mask2 = ma.where(y2>=y1)
    plt.bar(np.array([x for x in range(80)])[mask1], y1[mask1], color='red')
    plt.bar(np.array([x for x in range(80)]), y2, color='blue', label='pre-attack')
    plt.bar(np.array([x for x in range(80)])[mask2], y1[mask2], color='red', label='post-attack')
    plt.axhline(y = 0.5, color = 'black', linestyle = '-', label='treshold')
    plt.legend()
    plt.show()

################ LOAD PROFILE ################################

coefs = np.load('../experiment_results/{0}-{1}-profile.npy'.format(args.model_type, args.dataset_type))
epsilons = np.load('../experiment_results/{0}-{1}-profile-epsilons.npy'.format(args.model_type, args.dataset_type))

################ EXPERIMENT VARIABLES ########################

NUMBER_OF_SAMPLES = 100
max_eps = np.max(epsilons)
min_eps = np.min(max_eps) / 10
EPSILON_VALUES = [0.5*min_eps, min_eps, 2*min_eps, 4*min_eps, 6*min_eps, 8*min_eps, 10*min_eps]
print(EPSILON_VALUES)
flipped_labels = np.zeros((7, len(EPSILON_VALUES), NUMBER_OF_SAMPLES))
outputs  = np.zeros((5, len(EPSILON_VALUES), NUMBER_OF_SAMPLES, args.batch_size, args.num_classes))
targets = np.zeros((NUMBER_OF_SAMPLES, args.batch_size, args.num_classes))

# load, normalise the correlations and contruct inverted correlations
flipup_correlations = np.load('../experiment_results/flipup-correlations-cd-{0}-{1}.npy'.format(args.dataset_type, args.model_type))
flipup_correlations = flipup_correlations - np.min(flipup_correlations)
flipup_correlations = flipup_correlations / np.max(flipup_correlations)
flipdown_correlations = 1 - flipup_correlations

#############################  EXPERIMENT LOOP #############################

sample_count = 0

# DATASET LOOP
for i, (tensor_batch, labels) in enumerate(data_loader):
    tensor_batch = tensor_batch.to(device)

    if sample_count >= NUMBER_OF_SAMPLES:
        break

    # Do the inference
    with torch.no_grad():
        output = torch.sigmoid(model(tensor_batch))
        pred = (output > args.th).int()
        target = torch.clone(pred).detach()
        target = 1 - target
        targets[i, :, :] = target.cpu()

    # BUILD ICM
    negative_indices = np.where(target.cpu() == 0)[1]
    positive_indices = np.where(target.cpu() == 1)[1]
    instance_correlation_matrix = np.zeros(flipup_correlations.shape)
    instance_correlation_matrix[positive_indices] = flipup_correlations[positive_indices]
    instance_correlation_matrix[negative_indices] = flipdown_correlations[negative_indices]

    normalized_confidences = np.abs(output.cpu().numpy()) / np.max(np.abs(output.cpu().numpy()))

    # ATTACK LOOP
    for epsilon_index, epsilon in enumerate(EPSILON_VALUES):

        estimate = int(np.maximum(0, np.minimum(args.num_classes, poly.polyval(epsilon, coefs))))
        subset_length = int(np.minimum(args.num_classes, 1.66 * estimate))

        print('setlength =', subset_length)

        weights0 = get_top_n_weights(output, subset_length, random=True)
        weights1 = get_weights_from_correlations(instance_correlation_matrix, target, output, subset_length, 0, 4, 4)
        weights2 = get_weights_from_correlations(instance_correlation_matrix, target, output, subset_length, 0.5, 4, 4)
        weights3 = get_weights_from_correlations(instance_correlation_matrix, target, output, subset_length, 1, 4, 4)

        # PERFORM THE ATTACKS
        adversarials0 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(), eps=epsilon, device="cuda").detach()
        adversarials1 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=LinearLoss(), eps=epsilon, device="cuda").detach()
        adversarials2 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=SLAM(coefs, epsilon, max_eps, args.num_classes), eps=epsilon, device="cuda").detach()
        adversarials3 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(weight=weights0.to(device)), eps=epsilon, device="cuda").detach()
        adversarials4 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(weight=weights1.to(device)), eps=epsilon, device="cuda").detach()
        adversarials5 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(weight=weights2.to(device)), eps=epsilon, device="cuda").detach()
        adversarials6 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(weight=weights3.to(device)), eps=epsilon, device="cuda").detach()
        
        
        with torch.no_grad():

            # Another inference after the attack for adversarial predicition
            adv_output0 = torch.sigmoid(model(adversarials0))
            pred_after_attack0 = (adv_output0 > args.th).int()

            adv_output1 = torch.sigmoid(model(adversarials1))
            pred_after_attack1 = (adv_output1 > args.th).int()

            adv_output2 = torch.sigmoid(model(adversarials2))
            pred_after_attack2 = (adv_output2 > args.th).int()

            adv_output3 = torch.sigmoid(model(adversarials3))
            pred_after_attack3 = (adv_output3 > args.th).int()

            adv_output4 = torch.sigmoid(model(adversarials4))
            pred_after_attack4 = (adv_output4 > args.th).int()

            adv_output5 = torch.sigmoid(model(adversarials5))
            pred_after_attack5 = (adv_output5 > args.th).int()

            adv_output6 = torch.sigmoid(model(adversarials6))
            pred_after_attack6 = (adv_output6 > args.th).int()

            # store the outputs
            outputs[0, epsilon_index, i, :, :] = output.cpu()
            outputs[1, epsilon_index, i, :, :] = adv_output0.cpu()
            outputs[2, epsilon_index, i, :, :] = adv_output1.cpu()
            outputs[3, epsilon_index, i, :, :] = adv_output2.cpu()
            outputs[4, epsilon_index, i, :, :] = adv_output3.cpu()

            # store the flips        
            flipped_labels[0, epsilon_index, i*args.batch_size:(i+1)*args.batch_size] = torch.sum(torch.logical_xor(pred, pred_after_attack0), dim=1).cpu().numpy()
            flipped_labels[1, epsilon_index, i*args.batch_size:(i+1)*args.batch_size] = torch.sum(torch.logical_xor(pred, pred_after_attack1), dim=1).cpu().numpy()
            flipped_labels[2, epsilon_index, i*args.batch_size:(i+1)*args.batch_size] = torch.sum(torch.logical_xor(pred, pred_after_attack2), dim=1).cpu().numpy()
            flipped_labels[3, epsilon_index, i*args.batch_size:(i+1)*args.batch_size] = torch.sum(torch.logical_xor(pred, pred_after_attack3), dim=1).cpu().numpy()
            flipped_labels[4, epsilon_index, i*args.batch_size:(i+1)*args.batch_size] = torch.sum(torch.logical_xor(pred, pred_after_attack4), dim=1).cpu().numpy()
            flipped_labels[5, epsilon_index, i*args.batch_size:(i+1)*args.batch_size] = torch.sum(torch.logical_xor(pred, pred_after_attack5), dim=1).cpu().numpy()
            flipped_labels[6, epsilon_index, i*args.batch_size:(i+1)*args.batch_size] = torch.sum(torch.logical_xor(pred, pred_after_attack6), dim=1).cpu().numpy()
            
            # Confidence analysis plot
            # plot_confidences(output, adv_output0)

    sample_count += args.batch_size
    print('batch number:',i)

# SAVE THE RESULTS
np.save('../experiment_results/flips-{0}-{1}'.format(args.model_type, args.dataset_type), flipped_labels)
np.save('../experiment_results/maxdist-outputs-{0}-{1}'.format(args.model_type, args.dataset_type), outputs)
np.save('../experiment_results/maxdist-targets-{0}-{1}'.format(args.model_type, args.dataset_type), targets)

