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
from mlc_attack_losses import SLAM
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


def parse_model_and_args():

	########################## ARGUMENTS #############################################

	parser = argparse.ArgumentParser()

	parser.add_argument('classifier', type=str, default='asl_coco')
	parser.add_argument('data', metavar='DIR', help='path to dataset', default='coco')
	parser.add_argument('dataset_type', type=str, default='MSCOCO_2014')



	# IMPORTANT PARAMETERS!
	parser.add_argument('--th', type=float, default=0.5)
	parser.add_argument('-b', '--batch-size', default=1, type=int,
	                    metavar='N', help='mini-batch size (default: 16)')
	parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
	                    help='number of data loading workers (default: 16)')
	args = parse_args(parser)

	########################## SETUP THE MODELS  #####################


	if args.classifier == 'asl_coco':
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
	    q2l, config = create_q2l_model('q2l_coco.json')
	    args.model_type = 'q2l'
	    model = q2l

	elif args.classifier == 'q2l_nuswide':
	    q2l, config = create_q2l_model('q2l_nuswide.json')
	    args.model_type = 'q2l'
	    model = q2l

	args_dict = {**vars(args), **vars(config)}
	args = types.SimpleNamespace(**args_dict)

	return args, model


def load_dataset(args):

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

	return data_loader