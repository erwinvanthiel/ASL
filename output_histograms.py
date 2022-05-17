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
from attacks import pgd, fgsm, mi_fgsm, get_top_n_weights
from mlc_attack_losses import SigmoidLoss, HybridLoss, HingeLoss, LinearLoss, MSELoss, SmartLoss
from sklearn.metrics import auc
from src.helper_functions.helper_functions import mAP, CocoDetection, CocoDetectionFiltered, CutoutPIL, ModelEma, add_weight_decay
from src.helper_functions.voc import Voc2007Classification
from create_q2l_model import create_q2l_model
from src.helper_functions.nuswide_asl import NusWideFiltered
import numpy.polynomial.polynomial as poly
import numpy.ma as ma
import matplotlib as mpl
mpl.style.use('classic')


model = 'asl'
dataset = 'MSCOCO_2014'

outputs = np.load('experiment_results/maxdist-outputs-{0}-{1}.npy'.format(model, dataset))
targets = np.load('experiment_results/maxdist-targets-{0}-{1}.npy'.format(model, dataset))
flips = np.load('experiment_results/maxdist_epsilon-flips-{0}-{1}.npy'.format(model, dataset))

clean = outputs[0, 0, :, :, :]
baseline = outputs[1, 0, :, :, :] # 117 + 284 + 590 + 960 = 1951 fips
explicit_targets = outputs[2, 0, :, :, :] # 467 + 560 + 663 + 637 = 2327 flips
implicit_targets = outputs[3, 0, :, :, :] # 509 + 450 + 474 + 549 flips = 2027

clean[targets == 0] = 1 - clean[targets == 0]
baseline[targets == 0] = 1 - baseline[targets == 0]
explicit_targets[targets == 0] = 1 - explicit_targets[targets == 0]
implicit_targets[targets == 0] = 1 - implicit_targets[targets == 0]

print(np.sum(flips[0,0]))
print(np.sum(flips[1,0]))
print(np.sum(flips[2,0]))

plt.hist(clean.flatten(), bins = [i / 10 for i in range(0,10)])
plt.show()

plt.hist(baseline.flatten(), bins = [i / 10 for i in range(0,10)])
plt.show()

plt.hist(explicit_targets.flatten(), bins = [i / 10 for i in range(0,10)])
plt.show()

plt.hist(implicit_targets.flatten(), bins = [i / 10 for i in range(0,10)])
plt.show()