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
from attacks import pgd, fgsm, mi_fgsm, get_top_n_weights, l2_mi_fgm
from mlc_attack_losses import SigmoidLoss, HybridLoss, HingeLoss, LinearLoss, MSELoss
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
from src.helper_functions.voc import Voc2007Classification
from create_q2l_model import create_q2l_model
from src.helper_functions.nuswide_asl import NusWideFiltered

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

# # NUS_WIDE
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


