import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly

sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)

class SigmoidLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True, a=4):
        super(SigmoidLoss, self).__init__()
        self.a = a
        self.weight = weight

    def forward(self, x, y):        
        
        positive_loss = (-1 / (1 + torch.exp(-self.a*(x - 0.5)))+1)
        negative_loss = (1 / (1 + torch.exp(-self.a*(x - 0.5))))
        loss = y * positive_loss + (1-y) * negative_loss
        if self.weight is not None:
            loss = loss * self.weight
        loss = torch.mean(loss)
        return loss

class HybridLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True, a=4, t=0):
        super(HybridLoss, self).__init__()
        self.a = a
        self.t = t
        self.weight = weight

    def forward(self, x, y):        
        
        positive_loss = torch.maximum((-1 / (1 + torch.exp(-self.a*(x - 0.5 - self.t)))+1), -self.a*(x-self.t)*0.25 + self.a*0.125 + 0.5)
        negative_loss = torch.maximum((1 / (1 + torch.exp(-self.a*(x - 0.5 + self.t)))), self.a*(x+self.t)*0.25 - self.a*0.125 + 0.5)

        loss = y * positive_loss + (1-y) * negative_loss
        if self.weight is not None:
            loss = loss * self.weight
        loss = torch.mean(loss)
        return loss

class HingeLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super(HingeLoss, self).__init__()
        self.weight = weight

    def forward(self, x, y):        
        
        positive_loss = torch.maximum(0*x,0.5-x)
        negative_loss = torch.maximum(0*x,x-0.5)
        loss = y * positive_loss + (1-y) * negative_loss
        if self.weight is not None:
            loss = loss * self.weight
        loss = torch.mean(loss)
        return loss

class LinearLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super(LinearLoss, self).__init__()
        self.weight = weight

    def forward(self, x, y):        
        
        positive_loss = 1-x
        negative_loss = x
        loss = y * positive_loss + (1-y) * negative_loss
        if self.weight is not None:
            loss = loss * self.weight
        loss = torch.mean(loss)
        return loss

class MSELoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super(MSELoss, self).__init__()
        self.weight = weight

    def forward(self, x, y):        
        
        positive_loss = (1-x)**2
        negative_loss = x**2
        loss = y * positive_loss + (1-y) * negative_loss
        if self.weight is not None:
            loss = loss * self.weight
        loss = torch.mean(loss)
        return loss


class SmartLoss(nn.Module):
    
    def __init__(self, coefficients, epsilon, classifier_flip_distance, num_classes, weight=None, size_average=True):
        super(SmartLoss, self).__init__()
        
        if epsilon >= classifier_flip_distance:
            self.p = 1
        else:
            estimate = poly.polyval(epsilon, coefficients)
            self.p = np.minimum(1,(estimate / 80))
            print(epsilon, estimate, self.p)

        self.weight = weight

    def forward(self, x, y):
        bce = torch.nn.BCELoss(weight=self.weight)
        log_loss = bce(x,y)
        linear_loss = (y * (1-x) + (1-y) * x)

        if self.weight is not None:
            loss = loss * self.weight
        # loss_total = (1 - 0.5*self.p) * torch.mean(linear_loss) + 0.5 * self.p * log_loss
        loss_total = (1 - 0.5*self.p) * torch.mean(linear_loss) + 0.5*self.p * log_loss
        return loss_total
