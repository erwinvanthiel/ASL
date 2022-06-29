import argparse
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
import types
import torch
import torch.nn as nn
import _init_paths
from asl.src.helper_functions.helper_functions import parse_args
from asl.src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from asl.src.models import create_model

def get_args_from_dict(config):
	args = types.SimpleNamespace()	
	with open(config, 'r') as f:
		cfg_dict = json.load(f)
	for k,v in cfg_dict.items():
		if v == "True":
			v = True
		elif v == "False":
			v = False
		setattr(args, k, v)
	return args

def create_asl_model(config):
	args = get_args_from_dict(config)
	state = torch.load(args.model_path, map_location='cpu')
	asl = create_model(args).cuda()
	model_state = torch.load(args.model_path, map_location='cpu')
	asl.load_state_dict(model_state["state_dict"])
	return asl, args
	
