#-----------#
#  Imports  #
#-----------#
import torch
import torch.nn as nn # helps in creating and training the neural network
import torchvision
import torchvision.transforms as transforms # crop, resize, rotate, and vertically flip randomly
import matplotlib.pyplot as plt
import numpy as np
import statistics

#-------------------#
#  Hyperparameters  #
#-------------------#

#-------------------#
#  Device Settings  #
#-------------------#

#-------------------#
#  Dataset Loading  #
#-------------------#

#-----------#
#  Loaders  #
#-----------#

#----------------#
#  CNN Creation  #
#----------------#

#------------#
#  Accuracy  #
#------------#
def check_accuracy(loader, model):

    correct = 0
    total = 0

    model.eval() # evaluation mode

    # Don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    return float(correct) / float(total) * 100
#-----------#
#  Testing  #
#-----------#

#------------------#
#  Table Creation  #
#------------------#
