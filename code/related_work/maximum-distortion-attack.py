#@Time      :2019/12/15 16:16
#@Author    :zhounan
#@FileName  :attack_main_pytorch.py
import sys
sys.path.append('../')
from src.helper_functions.helper_functions import parse_args
import matplotlib
import torchvision.transforms as transforms
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import torch
import os
import numpy as np
import logging
from tqdm import tqdm
from attack_model import AttackModel
from create_q2l_model import create_q2l_model
from src.helper_functions.helper_functions import parse_args
from src.helper_functions.helper_functions import CocoDetectionFiltered

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # USE GPU

parser = argparse.ArgumentParser(description='multi-label attack')
parser.add_argument('--data', default='../../coco', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--dataset_type', default='MSCOCO_2014', type=str,
                    help='')
parser.add_argument('--image_size', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 32)')
parser.add_argument('--adv_batch_size', default=1, type=int,
                    metavar='N', help='batch size ml_cw, ml_rank1, ml_rank2 18, ml_lp 10, ml_deepfool is 10')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--adv_method', default='ml_deepfool', type=str, metavar='N',
                    help='attack method: ml_cw, ml_rank1, ml_rank2, ml_deepfool, ml_lp:97')
parser.add_argument('--adv_file_path', default='../data/voc2007/files/VOC2007/classification_mlgcn_adv.csv', type=str, metavar='N',
                    help='all image names and their labels ready to attack')
parser.add_argument('--adv_save_x', default='../adv_save/mlgcn/voc2007/', type=str, metavar='N',
                    help='save adversiral examples')
parser.add_argument('--adv_begin_step', default=0, type=int, metavar='N',
                    help='which step to start attacking according to the batch size')
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')


def new_folder(file_path):
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def init_log(log_file):
  new_folder(log_file)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

  fh = logging.FileHandler(log_file)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)

  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  ch.setFormatter(formatter)

  logger.addHandler(ch)
  logger.addHandler(fh)

def get_target_label(y, target_type):
    '''
    :param y: numpy, y in {0, 1}
    :param A: list, label index that we want to reverse
    :param C: list, label index that we don't care
    :return:
    '''
    y = y.copy()
    # o to -1
    y[y == 0] = -1
    if target_type == 'random_case':
        for i, y_i in enumerate(y):
            pos_idx = np.argwhere(y_i == 1).flatten()
            neg_idx = np.argwhere(y_i == -1).flatten()
            pos_idx_c = np.random.choice(pos_idx)
            neg_idx_c = np.random.choice(neg_idx)
            y[i, pos_idx_c] = -y[i, pos_idx_c]
            y[i, neg_idx_c] = -y[i, neg_idx_c]
    elif target_type == 'extreme_case':
        y = -y
    elif target_type == 'person_reduction':
        # person in 14 col
        y[:, 14] = -y[:, 14]
    elif target_type == 'sheep_augmentation':
        # sheep in 17 col
        y[:, 17] = -y[:, 17]
    elif target_type == 'hide_single':
        for i, y_i in enumerate(y):
            pos_idx = np.argwhere(y_i == 1).flatten()
            pos_idx_c = np.random.choice(pos_idx)
            y[i, pos_idx_c] = -y[i, pos_idx_c]
    return y

def gen_adv_file(model, target_type, adv_file_path):
    tqdm.monitor_interval = 0
    

    instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
    data_path = '{0}/val2014'.format(args.data)

    dataset = CocoDetectionFiltered(data_path,
                                instances_path,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))

    # Pytorch Data loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    output = []
    # image_name_list = []
    y = []
    test_loader = tqdm(test_loader, desc='Test')
    with torch.no_grad():
        nsamples = 100
        for i, (x, target) in enumerate(test_loader):
            if i >= nsamples:
                break
            if use_gpu:
                x = x.cuda()
            o = model(x).cpu().numpy()
            output.extend(o)
            y.extend(target.cpu().numpy())
            # image_name_list.extend(list(x[1]))
        output = np.asarray(output)
        y = np.asarray(y)
        # image_name_list = np.asarray(image_name_list)

    # choose x which can be well classified and contains two or more label to prepare attack
    pred = (output >= 0.5) + 0
    y[y==-1] = 0
    true_idx = []
    for i in range(len(pred)):
        if (y[i] == pred[i]).all() and np.sum(y[i]) >= 2:
            true_idx.append(i)
    # adv_image_name_list = image_name_list[true_idx]
    adv_y = y[true_idx]
    y = y[true_idx]
    y_target = get_target_label(adv_y, target_type)
    y_target[y_target==0] = -1
    y[y==0] = -1

    # print(len(adv_image_name_list))
    # adv_labeled_data = {}
    # for i in range(len(adv_image_name_list)):
    #     adv_labeled_data[adv_image_name_list[i]] = y[i]
    # write_object_labels_csv(adv_file_path, adv_labeled_data)

    # save target y and ground-truth y to prepare attack
    # value is {-1,1}


def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # set seed
    torch.manual_seed(1)
    if use_gpu:
        torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    # define dataset
    num_classes = 80

    # load torch model
    q2l = create_q2l_model('../config_coco.json')
    args.model_type = 'q2l'
    model = q2l

    if use_gpu:
        model = model.cuda()
   

    instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
    data_path = '{0}/val2014'.format(args.data)

    dataset = CocoDetectionFiltered(data_path,
                                instances_path,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))

    # Pytorch Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    state = {'model': model,
             'data_loader': data_loader,
             'adv_method': args.adv_method,
             'target_type': "extreme_case",
             'adv_batch_size': args.adv_batch_size,
             'y_target':0,
             'y': 0,
             'adv_save_x': 'adv/{0}/{1}/'.format('q2l',args.dataset_type),
             'adv_begin_step': args.adv_begin_step
             }

    # start attack
    attack_model = AttackModel(state)
    attack_model.attack()

    #evaluate_adv(state)
    #evaluate_model(model)

if __name__ == '__main__':
    main()