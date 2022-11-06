#-----------#
#  Imports  #
#-----------#
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#-------------------#
#  Hyperparameters  #
#-------------------#
input_size  = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10
#-------------------#
#  Device Settings  #
#-------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#-------------------#
#  Dataset Loading  #
#-------------------#
train_dataset = datasets.FashionMNIST (
    root = 'dataset/',
    train = True,
    transform = transforms.ToTensor(),
    download = True
)

train_loader = DataLoader (
    dataset = train_dataset, 
    batch_size = batch_size, 
    shuffle=True
    )

test_dataset = datasets.FashionMNIST (
    root = 'dataset/',
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

test_loader = DataLoader (
    dataset = train_dataset, 
    batch_size = batch_size, 
    shuffle=True
)
#----------------#
#  CNN Creation  #
#----------------#
class NN(nn.Module):
  def __init__(self, input_size, num_classes):
    super (NN, self).__init__()
    self.fc1 = nn.Linear (input_size, 50)
    self.fc2 = nn.Linear (50, num_classes)

  def forward (self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
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
model = NN(
    input_size=input_size,
    num_classes = num_classes
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam (model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  for batch_idx, (data, targets) in enumerate(train_loader):
    
    # get data to cud if possible
    data = data.to(device=device)
    targets = targets.to(device=device)

    # Get to correct shape
    data = data.reshape(data.shape[0], -1)

    # Forward
    scores = model(data)
    loss = criterion(scores, targets)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Gradient desecent or adam step
    optimizer.step()
    
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
#------------------#
#  Table Creation  #
#------------------#
