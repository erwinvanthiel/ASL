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
from attacks import pgd, fgsm, mi_fgsm, unrestricted_mi_fgsm, get_weights_from_correlations, generate_subset, objective_function
from mlc_attack_losses import SigmoidLoss, HybridLoss, HingeLoss, LinearLoss, MSELoss
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
from src.helper_functions.voc import Voc2007Classification
from create_q2l_model import create_q2l_model
from src.helper_functions.nuswide_asl import NusWideFiltered
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)

########################## ARGUMENTS #############################################

parser = argparse.ArgumentParser()

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

# # # NUS_WIDE
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
q2l = create_q2l_model('config_coco.json')
args.model_type = 'q2l'
model = q2l



# LOAD THE DATASET WITH DESIRED FILTER

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


################ EXPERIMENT VARIABLES ########################

NUMBER_OF_SAMPLES = 100
numbers_of_levels = [2] #[x+1 for x in range(10)]
numbers_of_branches = [3]#[x+1 for x in range(10)]
gamma_values = [0.5]
numbers_of_labels = [10]
objective_values = np.zeros((len(gamma_values), len(numbers_of_labels), len(numbers_of_levels), len(numbers_of_branches), NUMBER_OF_SAMPLES))

# load, normalise the correlations and contruct inverted correations
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

    sample_count += 1

    # Do the inference
    with torch.no_grad():
        output = torch.sigmoid(model(tensor_batch))
        pred = (output > args.th).int()
        target = torch.clone(pred).detach()
        target = (1 - target).cpu().numpy()
        output = output.cpu().numpy()
    
    negative_indices = np.where(target == 0)[1]
    positive_indices = np.where(target == 1)[1]

    instance_correlation_matrix = np.zeros(flipup_correlations.shape)
    instance_correlation_matrix[positive_indices] = flipup_correlations[positive_indices]
    instance_correlation_matrix[negative_indices] = flipdown_correlations[negative_indices]
    

    normalized_confidences = np.abs(output) / np.max(np.abs(output))

    for gamma_id, gamma in enumerate(gamma_values):

        for number_of_labels_id, number_of_labels in enumerate(numbers_of_labels):
            for numbers_of_levels_id, number_of_levels in enumerate(numbers_of_levels):
                for number_of_branches_id, number_of_branches in enumerate(numbers_of_branches):

                    if number_of_branches ** number_of_levels > 10000:
                        objective_values[gamma_id, number_of_labels_id, numbers_of_levels_id, number_of_branches_id, i] = 0
                    else:
                        subset = generate_subset(output, instance_correlation_matrix, number_of_labels, gamma, number_of_branches, number_of_levels)
                        value = objective_function(subset, instance_correlation_matrix, normalized_confidences, gamma)
                        objective_values[gamma_id, number_of_labels_id, numbers_of_levels_id, number_of_branches_id, i] = value
                
    print(i)

np.save('experiment_results/tree-depth-x-branches-{0}-{1}.npy'.format(args.model_type, args.dataset_type), objective_values)