#just a rough code architecture using pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
       
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=, out_channels=, kernel_size=, stride=, padding=),
            nn.ReLU(),
            nn.MaxPool1d()
        )
       
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=, out_channels=, kernel_size=, stride=, padding=),
            nn.ReLU(),
            nn.MaxPool1d()
        )
       
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=, out_channels=, kernel_size=, stride=, padding=),
            nn.ReLU(),
            nn.MaxPool1d()
        )
       
        self.fc1 = nn.Linear(in_features=, out_features=)
       
        self.fc2 = nn.Linear(in_features=, out_features=)
       
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
       
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = CNN()
