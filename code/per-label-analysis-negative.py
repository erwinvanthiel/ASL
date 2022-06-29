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
epsilon = 0.008

flipped_labels = np.zeros((args.num_classes, NUMBER_OF_SAMPLES))
actual_samples = np.zeros((args.num_classes))
confidences = np.zeros((args.num_classes, NUMBER_OF_SAMPLES))

for target_label in range(args.num_classes):

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
                                    ]), label_indices_positive=np.array([target_label]))
    elif args.dataset_type == 'PASCAL_VOC2007':

        dataset = Voc2007Classification('trainval',
                                        transform=transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                    ]), train=True, label_indices_positive=np.array([target_label]))

    elif args.dataset_type == 'NUS_WIDE':
        
        dataset = NusWideFiltered('train', transform=transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor()]), label_indices_positive=np.array([target_label])
        )

    # Pytorch Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)



    #############################  EXPERIMENT LOOP #############################

    sample_count = 0
    batch_number = 0

    # DATASET LOOP
    for idx, (tensor_batch, labels) in enumerate(data_loader):
        tensor_batch = tensor_batch.to(device)

        if sample_count >= NUMBER_OF_SAMPLES:
            break


        # Do the inference
        with torch.no_grad():
            output = torch.sigmoid(model(tensor_batch))
            pred = (output > args.th).int()
            target = torch.clone(pred).detach()
            target = 1 - target
            weights = torch.zeros(target.shape)
            weights[:, target_label] = 1 

        if pred[:,target_label].sum() < args.batch_size:
            continue

        adversarials = mi_fgsm(model, tensor_batch.detach(), target, loss_function=torch.nn.BCELoss(weight=weights.to(device)), eps=epsilon, device="cuda").detach()
        
        with torch.no_grad():
        
            pred_after_attack = (torch.sigmoid(model(adversarials)) > args.th).int()
            flipped_labels[target_label, batch_number*args.batch_size:(batch_number+1)*args.batch_size] = args.batch_size - pred_after_attack[:, target_label].cpu()
            confidences[target_label, batch_number*args.batch_size:(batch_number+1)*args.batch_size] = output[:, target_label].cpu()
            
        print('batch number:',batch_number)
        sample_count += args.batch_size
        batch_number += 1
        actual_samples[target_label] += 1
        


np.save('experiment_results/positive-label-flips-actual-samples-{0}-{1}'.format(args.model_type, args.dataset_type), actual_samples)
np.save('experiment_results/positive-label-flips-{0}-{1}'.format(args.model_type, args.dataset_type), flipped_labels)
np.save('experiment_results/positive-label-confidences-{0}-{1}'.format(args.model_type, args.dataset_type), confidences)





