import os
import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib
import torchvision.transforms as transforms
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from attacks import pgd, fgsm, mi_fgsm, get_top_n_weights, get_weights_from_correlations
from mlc_attack_losses import SigmoidLoss, HybridLoss, HingeLoss, LinearLoss, MSELoss, SmartLoss
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
from src.helper_functions.voc import Voc2007Classification
from create_q2l_model import create_q2l_model
from src.helper_functions.nuswide_asl import NusWideFiltered
import numpy.polynomial.polynomial as poly
import numpy.ma as ma
import matplotlib as mpl
import numpy.polynomial.polynomial as poly
mpl.style.use('classic')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)

########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser()

# MSCOCO 2014
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
# parser.add_argument('--model_path', type=str, default='./models/tresnetl-asl-mscoco-epoch80')
# parser.add_argument('--model_name', type=str, default='tresnet_l')
# parser.add_argument('--num-classes', default=80)
# parser.add_argument('--dataset_type', type=str, default='MSCOCO_2014')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# PASCAL VOC2007
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='../VOC2007')
# parser.add_argument('attack_type', type=str, default='PGD')
# parser.add_argument('--model-path', default='./models/tresnetxl-asl-voc-epoch80', type=str)
# parser.add_argument('--model_name', type=str, default='tresnet_xl')
# parser.add_argument('--num-classes', default=20)
# parser.add_argument('--dataset_type', type=str, default='PASCAL_VOC2007')
# parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')

# # NUS_WIDE
parser.add_argument('data', metavar='DIR', help='path to dataset', default='../NUS_WIDE')
parser.add_argument('--model_path', type=str, default='./models/tresnetl-asl-nuswide-epoch80')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--num-classes', default=81)
parser.add_argument('--dataset_type', type=str, default='NUS_WIDE')
parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')


# IMPORTANT PARAMETERS!
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
args = parse_args(parser)



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
# q2l = create_q2l_model('config_nuswide.json')
# args.model_type = 'q2l'
# model = q2l



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

################ LOAD PROFILE ################################

coefs = np.load('experiment_results/{0}-{1}-profile.npy'.format(args.model_type, args.dataset_type))
epsilons = np.load('experiment_results/{0}-{1}-profile-epsilons.npy'.format(args.model_type, args.dataset_type))

################ EXPERIMENT VARIABLES ########################

NUMBER_OF_SAMPLES = 100
max_eps = np.max(epsilons)
min_eps = np.min(max_eps) / 10
EPSILON_VALUES = [0.5*min_eps, min_eps, 2*min_eps, 4*min_eps, 6*min_eps, 8*min_eps, 10*min_eps]
# print(EPSILON_VALUES)
flipped_labels = np.zeros((4, len(EPSILON_VALUES), NUMBER_OF_SAMPLES))
outputs  = np.zeros((5, len(EPSILON_VALUES), NUMBER_OF_SAMPLES, args.batch_size, args.num_classes))
targets = np.zeros((NUMBER_OF_SAMPLES, args.batch_size, args.num_classes))

# load, normalise the correlations and contruct inverted correlations
flipup_correlations = np.load('experiment_results/flipup-correlations-cd-{0}-{1}.npy'.format(args.dataset_type, args.model_type))
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
        # targets[i, :, :] = target.cpu()

    negative_indices = np.where(target.cpu() == 0)[1]
    positive_indices = np.where(target.cpu() == 1)[1]

    instance_correlation_matrix = np.zeros(flipup_correlations.shape)
    instance_correlation_matrix[positive_indices] = flipup_correlations[positive_indices]
    instance_correlation_matrix[negative_indices] = flipdown_correlations[negative_indices]

    normalized_confidences = np.abs(output.cpu().numpy()) / np.max(np.abs(output.cpu().numpy()))

    # process a batch and add the flipped labels for every number of targets
    for epsilon_index, epsilon in enumerate(EPSILON_VALUES):

        estimate = int(np.maximum(0, np.minimum(args.num_classes, poly.polyval(epsilon, coefs))))
        subset_length = int(np.minimum(args.num_classes, 1.66 * estimate))
        print('setlength =', subset_length)

        # adversarials0 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(), eps=epsilon, device="cuda").detach()
        # adversarials1 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=LinearLoss(), eps=epsilon, device="cuda").detach()
        # adversarials2 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=SmartLoss(coefs, epsilon, max_eps, args.num_classes), eps=epsilon, device="cuda").detach()


        weights0 = get_top_n_weights(output, subset_length, random=True)
        weights1 = get_weights_from_correlations(instance_correlation_matrix, target, output, subset_length, 0, 4, 4)
        weights2 = get_weights_from_correlations(instance_correlation_matrix, target, output, subset_length, 0.5, 4, 4)
        weights3 = get_weights_from_correlations(instance_correlation_matrix, target, output, subset_length, 1, 4, 4)



        adversarials0 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(weight=weights0.to(device)), eps=epsilon, device="cuda").detach()
        adversarials1 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(weight=weights1.to(device)), eps=epsilon, device="cuda").detach()
        adversarials2 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(weight=weights2.to(device)), eps=epsilon, device="cuda").detach()
        adversarials3 = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(weight=weights3.to(device)), eps=epsilon, device="cuda").detach()
        
        
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

            # Confidence analysis plot
            # plot_confidences(output, adv_output0)

    sample_count += args.batch_size
    print('batch number:',i)

# flipped_labels = np.insert(flipped_labels, 0, 0, axis=2)
# EPSILON_VALUES.insert(0,0)

# means_bce = np.mean(flipped_labels,axis=2)[0]
# means_linear = np.mean(flipped_labels,axis=2)[1]
# means_smart = np.mean(flipped_labels,axis=2)[2]

# std_bce = np.std(flipped_labels,axis=2)[0]
# std_linear = np.std(flipped_labels,axis=2)[1]
# std_smart = np.std(flipped_labels,axis=2)[2]

np.save('experiment_results/explicit-flips-{0}-{1}'.format(args.model_type, args.dataset_type), flipped_labels)
# np.save('experiment_results/maxdist-outputs-{0}-{1}'.format(args.model_type, args.dataset_type), outputs)
# np.save('experiment_results/maxdist-targets-{0}-{1}'.format(args.model_type, args.dataset_type), targets)

