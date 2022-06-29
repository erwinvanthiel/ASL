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

########################## LOAD MODEL #############################################

args, model = parse_model_and_args()

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

################ EXPERIMENT VARIABLES  ########################

NUMBER_OF_SAMPLES = 100
EPSILON_VALUES = [0.005]

#############################  EXPERIMENT LOOP #############################

correlations = torch.zeros((len(EPSILON_VALUES),args.num_classes,args.num_classes))
confidences = torch.zeros((len(EPSILON_VALUES),args.num_classes,args.num_classes))

sample_count = 0

for i, (tensor_batch, labels) in enumerate(data_loader):
    tensor_batch = tensor_batch.to(device)

    if sample_count >= NUMBER_OF_SAMPLES:
        break

    for epsilon_index, epsilon in enumerate(EPSILON_VALUES):

        for target_label in range(args.num_classes):

            target = torch.zeros(args.batch_size, args.num_classes).to(device).float()
            target[:, target_label] = 1
            loss_weights = torch.zeros(target.shape).to(device)
            loss_weights[:, target_label] = 1

            adversarials = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(weight=loss_weights.detach()), eps=epsilon, device='cuda').detach()            

            with torch.no_grad():
                correlations[epsilon_index, target_label] += (1 / NUMBER_OF_SAMPLES) * (torch.sigmoid(model(adversarials)) - torch.sigmoid(model(tensor_batch))).sum(dim=0).cpu()
                # confidences[epsilon_index, target_label] += torch.sigmoid(model(adversarials)).sum(dim=0).cpu()
        
        ## Code for finding suitable epsilon ###
        # target_label_confidences = []
        # for p in range(args.num_classes):
        #     target_label_confidences.append(confidences[epsilon_index,p,p].item())
        # print(epsilon, np.min(target_label_confidences), np.mean(target_label_confidences) * NUMBER_OF_SAMPLES)

    sample_count += args.batch_size
    print(sample_count)



# save the results
np.save('experiment_results/flipup-correlations-{0}-{1}.npy'.format(args.dataset_type, args.model_type), correlations[0])
sns.heatmap(correlations[0])
plt.show()
