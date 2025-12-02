#just a rough code architecture using pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

#light curve data
class LightCurveDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path)
        self.y = np.load(y_path)
        self.x = self.x.reshape(-1, 1, self.x.shape[1]).astype(np.float32)
        self.y = self.y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


#CNN model
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
    
    
#training_hyperparameters
learning_rate = 0.001
num_epochs = 100
batch_size = 128
train_size = 0.8
val_size = 0.1
    
#load data
dataset = LightCurveDataset(x_path = "x_data.npy", y_path = "y_data.npy")
n = len(dataset)
num_train = int(n * train_size)
num_val = int(n * val_size)
num_test = n - num_train - num_val

train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)


#training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#training loop
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = 
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

