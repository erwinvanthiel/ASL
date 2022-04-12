import os
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import auc


t1 = torch.ones((2,3,448,448))
t2 = torch.ones((2,3,448,448)) * 3
linf = torch.amax(torch.abs(t1 - t2), (1,2,3))
print(linf)