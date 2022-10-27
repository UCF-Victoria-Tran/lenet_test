# Importing the libraries
import torch
import torch.nn as nn # helps in creating and training the neural network
import torchvision
import torchvision.transforms as transforms # crop, resize, rotate, and vertically flip randomly
import matplotlib.pyplot as plt
import numpy as np
import statistics

# Define relevant variables for the machine learning task
batch_size = 64  # number of samples that will be passed through the network at one time
num_classes = 10 # targets/labels or categories, list of data set allocation attributes and their values
learning_rate = 0.001 # configurable hyperparameter, small positive value 0.0 <-> 1.0, controls how quickly model adapted to problem
num_epochs = 10 # number of complete passes of the entire training dataset passing through the training or learning process of the algorithm

# Device will run training on CPU
device = torch.device('cpu')
# If you have CUDA installed you can use this instead:
# device = torch.device('cuda')
# If you don't know if you have CUDA or not you can use this:
# device = torch.device('cuda' if torch.cuda.is_avaliable() else 'cpu')

# Loading the dataset and preprocessing
# ROOT offers naive support for supervised learning techniques, such as multivariate
# classification (both binary and multi class) and regression. Also allows easy
# interoperability with commonly used machine learning libraries.
train_dataset = torchvision.datasets.MNIST(
    root = './data',
    train = True, # the training dataset is true
    transform = transforms.Compose([
        transforms.Resize((32,32)), # Making a transform that will resize the images and crop them differently but still get output of 32x32 to put in LeNet
        transforms.ToTensor(), # Getting the output as a 'tensor', basically imagine a matrix (multi-dimensional array) of numbers
        transforms.Normalize(mean = (0.1307,), std = (0.3081,))
        # changing the values of numeric columns in the dataset to use a common scale, without distorting difference in the ranges or values or losing info
        # improves performance and training stability of the model, like getting an average instead of extremes on either side (outliers)
        # not every dataset requires normalization, only required when features have different ranges
    ]),
    download = True # incase the data is not already downloaded
)

test_dataset = torchvision.datasets.MNIST(
    root = './data',
    train = False,
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1325,), std = (0.3105,))
    ]),
    download = True
)

# Data loaders allow us to iterate through the data in batches, and the data is loaded while iterating and
# not at once in the start.
# Can impede the performance in case of large datasets, but with a small dataset like MNIST it's fine
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = True
)

# Defining the convolutional neural network
# Creating a class that inherits from nn.Module as it contains many methods we need to utilize
class LeNet5(nn.Module):
    # Creating the CNN itself
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            # Kernel = 2D array of weights, Filter = multiple kernels stacked together. In 2D they are the same.
            # first parameter is in_channels, secound is out_channels. Can leave out the names
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    # Sequence in which the layers will process the image
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# Setting some hyperparameters, such as the loss function and the optimizer to be used
model = LeNet5(num_classes).to(device)

# Setting the loss function, sometimes called 'criterion'
cost = nn.CrossEntropyLoss()

#Setting the optimizer with the model parameters and learning rate
# Adam = Adaptive Movement Estimation, an algorithm for optimization techique for gradient descent
# The method is really efficient when working with large problems involving a lot of sata or parameters
# Requires less memory and is efficient
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Print how many steps are remaining when training
# To better keep track of steps when training
total_step = len(train_loader)

# Making points for plotting
loss_values = []
loss_median = []

# Training the model!
# iterating through the number of epochs then the batches in our training data
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): # convert images and labels according to device we are using
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        # make predictions using our model and calculate loss based on those predictions and our actual labels
        outputs = model(images)
        loss = cost(outputs, labels)

        # getting data points through every training loop and putting in list
        loss_values.append(loss.item())

        # backward and optimize
        # actually update our weights to improve our model
        optimizer.zero_grad() # set gradients to zero before every update
        loss.backward() # calculate new gradients
        optimizer.step() # update weights

        if (i+1) % 500 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    loss_median.append(statistics.median(loss_values))

# Test the model!
# In test phase, don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'
        .format(100 * correct / total))

# Loss = number indicating how bad the model's prediction was on a single example.
# model's prediction is perfect = loss is zero. Otherwise, loss is greater.
plt.plot(loss_median, label = 'Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()
