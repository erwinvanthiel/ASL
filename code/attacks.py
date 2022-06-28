import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D 
import logging
import mosek
import gc
from multiprocessing import Pool
from mlc_attack_losses import LinearLoss
import math
import matplotlib.pyplot as plt
import seaborn as sns


sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)


def pgd(model, images, target, loss_function=torch.nn.BCELoss(), eps=0.3, alpha=2/255, iters=10, device='cuda'):

    loss = loss_function
    images = images.to(device).detach()
    target = target.to(device).float().detach()
    model = model.to(device)
    ori_images = images.data.to(device)

    for i in range(iters):    
        images.requires_grad = True

        # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!
        outputs = sigmoid(model(images)).to(device)
        model.zero_grad()
        cost = 0

        if target_ids:
            cost = loss(outputs[:, target_ids], target[:, target_ids].detach())
        else:
            cost = loss(outputs, target)

        cost.backward()

        # perform the step
        adv_images = images - alpha * images.grad.sign()
        # print(images.grad[0])

        # bound the perturbation
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        # construct the adversarials by adding perturbations
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images


# Momentum Induced Fast Gradient Sign Method 
def mi_fgsm(model, images, target, loss_function=torch.nn.BCELoss(), eps=0.3, device='cuda'):
    
    # put tensors on the GPU
    images = images.to(device)
    target = target.to(device).float()
    model = model.to(device)

    L = loss_function

    alpha = (1/256)/10
    iters = int(eps / alpha)
    # iters = 10
    # alpha = eps / iters
    mu = 1.0
    g = 0
    
    for i in range(iters):    
        images.requires_grad = True

        # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!


        outputs = sigmoid(model(images)).to(device)


        model.zero_grad()
        cost = L(outputs, target.detach())
        cost.backward()

        # normalize the gradient
        new_g = images.grad / torch.sum(torch.abs(images.grad))

        # update the gradient
        g = mu * g + new_g

        # perform the step, and detach because otherwise gradients get messed up.
        images = (images - alpha * g.sign()).detach()

    # clamp the output
    images = torch.clamp(images, min=0, max=1).detach()
            
    return images

# Momentum Induced Fast Gradient Sign Method 
def l2_mi_fgm(model, images, target, loss_function=torch.nn.BCELoss(), eps=0.3, device='cuda'):
    
    # put tensors on the GPU
    images = images.to(device)
    starting_points = images.clone()
    target = target.to(device).float()
    model = model.to(device)

    L = loss_function
    inner_cube_width = 2 * eps
    hyper_sphere_radius = (inner_cube_width * 187.76)
    hyper_sphere_radius = eps

    alpha = float(448 / 2560)
    iters = int(hyper_sphere_radius / alpha)
    mu = 1.0
    g = 0
    
    for i in range(iters):    
        images.requires_grad = True

        outputs = sigmoid(model(images)).to(device)

        model.zero_grad()
        cost = L(outputs, target.detach())
        cost.backward()

        # normalize the gradient
        new_g = images.grad / torch.sqrt(torch.sum(images.grad ** 2))

        # update the gradient
        g = mu * g + new_g

        # perform the step, and detach because otherwise gradients get messed up.
        images = (images - alpha * g).detach()
    

    # clamp the output
    images = torch.clamp(images, min=0, max=1).detach()
            
    return images    


def get_top_n_weights(outputs, number_of_attacked_labels, random=False):
    # rankings = (1-target_vector) * outputs + target_vector * (1-outputs)
    rankings = torch.abs(outputs - 0.5)
    rankings = torch.argsort(rankings, dim=1, descending=False)
    weights = torch.zeros(outputs.shape)
    if random == True:
        weights[:, np.random.permutation(outputs.shape[1])[0:number_of_attacked_labels]] = 1
    else:
        weights[:, rankings[:, 0:number_of_attacked_labels]] = 1
    return weights


def get_weights_from_correlations(correlations, target, outputs, number_of_labels, gamma, number_of_branches, branch_depth):

    weights = torch.zeros(target.shape)
    outputs = (target) * outputs + (1-target) * (1-outputs)
    outputs = outputs.detach().cpu().numpy()
    target = target.cpu().numpy()

    for i in range(target.shape[0]):
        weights[i, generate_subset(outputs[i, :], correlations, number_of_labels, gamma, number_of_branches, branch_depth)] = 1
    return weights

        
class TreeOfLists():

    def __init__(self):
        self.baselist = []
        self.added_labels = []
        self.children = []

    def add_child(self, label):
        child = TreeOfLists()
        child.baselist = self.baselist.copy()
        child.added_labels = self.added_labels.copy()
        child.added_labels.append(label)
        self.children.append(child)

    def get_list(self):
        total_list = self.baselist.copy() + self.added_labels.copy()
        return total_list

# Classifier Landscape Attentive Subset Selection (CLASS)
def generate_subset(outputs, instance_correlation_matrix, number_of_labels, gamma, number_of_branches, branch_depth):

    
    normalized_confidences = np.squeeze(np.abs(outputs) / np.max(np.abs(outputs)))

    confidence_rankings = np.squeeze(np.argsort(normalized_confidences))
    root_label = confidence_rankings[len(confidence_rankings) - 1].item()

    # Initialize the label set with easiest/closest label
    base_label_set = [root_label]

    # We iteratively add a label until pre-specified length is reached
    for l in range(number_of_labels-1):

        # We have 'number_of_branches' branches to explore up until depth 'branch_depth' for the best option
        root = TreeOfLists()
        root.baselist = base_label_set.copy()
        parents = [root]
        children = []
        depth = min(branch_depth, number_of_labels - len(base_label_set))

        # Look mutiple levels ahead and pick the best option to add to the list
        for d in range(depth):

            for parent in parents:
                # print(parent.get_list())
                current_label_set = parent.get_list()

                ## COMPUTE THE CURRENT LABEL RANKINGS FOR THIS PARENT

                # We compute the correlations from and to the set by using the correlation matrix, we then select the best option 
                correlation_to_set = instance_correlation_matrix[:, current_label_set].sum(axis=1)
                correlation_from_set = instance_correlation_matrix[current_label_set, :].sum(axis=0)
                correlation_factors = correlation_to_set + correlation_from_set
                # normalized_correlation_factors = (correlation_factors-np.min(correlation_factors)) / np.max(correlation_factors-np.min(correlation_factors))

                # gamma determines the priority distribution between label confidence and correlation
                scores = gamma * correlation_factors + (1-gamma) * normalized_confidences
                ranking = np.squeeze(np.argsort(scores))
                updated_ranking = [x for x in ranking if x not in current_label_set]
                ## FOR EACH BRANCH ADD A TOP LABEL FROM THE RANKING
                for b in range(min(number_of_branches, len(updated_ranking))):
                    added_label = updated_ranking[len(updated_ranking)-1-b]
                    parent.add_child(added_label)
                children.extend(parent.children)
            # print([c.get_list() for c in children])
            parents = children
            children = []

        # find the best leaf node and use its parent from the first sub-root level as a next added label
        max_obj_value = 0
        best_option = None
        for p in parents:
            obj_value = objective_function(p.get_list(), instance_correlation_matrix, normalized_confidences, gamma)
            # print(p.get_list(), obj_value)
            if obj_value > max_obj_value:
                max_obj_value = obj_value
                best_option = p
        base_label_set.append(best_option.added_labels[0])

    return base_label_set


def objective_function(label_set, instance_correlation_matrix, normalized_confidences, gamma):
    correlation_score = 0
    for label in label_set:
        correlation_score = correlation_score + instance_correlation_matrix[label, label_set].sum()
    confidence_score = np.squeeze(normalized_confidences)[label_set].sum()
    return gamma * correlation_score + (1-gamma) * confidence_score


# Fast Gradient Sign Method 
def fgsm(model, images, target, loss_function=torch.nn.BCELoss(), eps=0.3, device='cuda'):
    
    # put tensors on the GPU
    images = images.to(device).detach()
    target = target.to(device).float()
    model = model.to(device)
    loss = nn.BCELoss()
    images.requires_grad = True

    # USE SIGMOID FOR MULTI-LABEL CLASSIFIER!
    outputs = sigmoid(model(images)).to(device)

    # Compute loss and perform back-prop
    model.zero_grad()
    cost = loss(outputs, target)
    cost.backward()

    # perform the step
    images = images - eps * images.grad.sign()

    # clamp the output
    images = torch.clamp(images, min=0, max=1)

    return images


    