import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

#path to the data
#assume this data dir pipeline_output_final/final_metadata.csv and pipeline_output_final/batch_*.npz are in data_dir
data_dir = "pipeline_output_final"

#===============
#dataset class
#===============
#this is class which is for V4 preprocessing data. You need to have final_metadata.csv and batch_*.npz in data_dir 
class ExoplanetData(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory containing the .npz batch files
        """
        self.data_dir = data_dir
        self.meta = pd.read_csv(os.path.join(data_dir, "final_metadata.csv"))
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        """
        Returns:
            global_x: Global view flux data (torch.Tensor, shape: (1, length))
            local_x: Local view flux data (torch.Tensor, shape: (1, length))
            label: Label (torch.Tensor, dtype: float32)
        """
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.data_dir, row['filename'])
        
        with np.load(file_path) as data:
            index = int(row['index_in_batch'])  # Convert to int for array indexing
            flux_local = data['flux_local'][index].astype(np.float32).reshape(1, -1)
            flux_global = data['flux_global'][index].astype(np.float32).reshape(1, -1)
            label = int(row['label'])
            
        return (
            torch.from_numpy(flux_global).float(), 
            torch.from_numpy(flux_local).float(), 
            torch.tensor(label, dtype=torch.float32)
            )
    

#===============
#CNN model
#===============
#CNN model:two disjoinit conv colomns(local and global), after that single MLP layer
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
       
       #for local cnn column(input is (1, 201))
        self.local_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7, stride=2)
        )
        #output_shape: (16, 94)
       
       
       #don't use for small current model
        #self.local_conv2 = nn.Sequential(
        #    nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5),
        #    nn.ReLU(),
        #    nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5),
        #    nn.ReLU(),
        #    nn.MaxPool1d(kernel_size=7, stride=2)
        #)
        #output_shape: (32, 40)
        
        
        #for global cnn column(input is (1, 2001))
        self.global_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2)
        )
        #output shape: (16, 995)
        
        
        self.global_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2)
        )
        #output_shape: (32, 492)
        
        #don't use for small current model
        #self.global_conv3 = nn.Sequential(
        #    nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5),
        #    nn.ReLU(),
        #    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
        #    nn.ReLU(),
        #    nn.MaxPool1d(kernel_size=5, stride=2)
        #)
        #output_shape: (64, 240)
        
        #don't use for small current model
        #self.global_conv4 = nn.Sequential(
        #    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5),
        #    nn.ReLU(),
        #    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5),
        #    nn.ReLU(),
        #    nn.MaxPool1d(kernel_size=5, stride=2)
        #)
        #output_shape: (128, 114)
        
        #don't use for small current model
        #self.global_conv5 = nn.Sequential(
        #    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5),
        #    nn.ReLU(),
        #    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5),
        #    nn.ReLU(),
        #    nn.MaxPool1d(kernel_size=5, stride=2)
        #)
        #output_shape: (256, 51)
        
        #for mlp layer
        self.fc1 = nn.Linear(in_features=16*94 + 32*492, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
       
    def forward(self, global_x, local_x):
        
        #local cnn column
        local_x = self.local_conv1(local_x)
        local_x = torch.flatten(local_x, 1)
        
        
        
        #global cnn column
        global_x = self.global_conv1(global_x)
        global_x = self.global_conv2(global_x)
        global_x = torch.flatten(global_x, 1)
        
        
        #mlp layer
        x = torch.cat((local_x, global_x), 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
#===============
#training hyperparameters
#===============
#training_hyperparameters
learning_rate = 0.001
num_epochs = 100
batch_size = 64
train_size = 0.8
val_size = 0.1

#===============
#split data into train, validation, and test
#===============
#split data into train, validation, and test
dataset = ExoplanetData(data_dir = data_dir)
n = len(dataset)
num_train = int(n * train_size)
num_val = int(n * val_size)
num_test = n - num_train - num_val

train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

#===============
#training setup
#===============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


train_losses = []
train_accs = []
val_losses = []
val_accs = []

#===============
#training loop
#===============
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_acc = 0
    for global_x, local_x, y in train_loader:

        global_x, local_x, y = global_x.to(device), local_x.to(device), y.to(device)
        pred = model(global_x, local_x) 
        prob = torch.sigmoid(pred).view(-1)
        loss = criterion(pred.view(-1), y)
        
        #backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        pred_label = (prob > 0.5).float()    
        train_loss += loss.item() * y.size(0)
        train_acc += (pred_label == y).float().sum().item()
        
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    
    #store history
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
        
    #validation
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for global_x, local_x, y in val_loader:
            global_x, local_x, y = global_x.to(device), local_x.to(device), y.to(device)
            
            pred = model(global_x, local_x)
            prob = torch.sigmoid(pred).view(-1)
            loss = criterion(pred.view(-1), y)
            
            pred_label = (prob > 0.5).float()    
            val_loss += loss.item() * y.size(0)
            val_acc += (pred_label == y).float().sum().item()
            
    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)
    
    #store history
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    #print progress
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    
#===============
#plot results
#===============
epochs = range(1, num_epochs+1)
plt.figure(figsize=(12, 5))

#plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'r', label='Training loss')
plt.plot(epochs, val_losses, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

#plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, 'r', label='Training accuracy')
plt.plot(epochs, val_accs, 'b', label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
    
    
    
    
#===============
#evaluation function
#===============
def evaluate(pred_labels, y_list):
    pred_labels =torch.cat(pred_labels)
    y_list = torch.cat(y_list)
    TP = ((pred_labels == 1) & (y_list == 1)).float().sum().item()
    FP = ((pred_labels == 1) & (y_list == 0)).float().sum().item()
    TN = ((pred_labels == 0) & (y_list == 0)).float().sum().item()
    FN = ((pred_labels == 0) & (y_list == 1)).float().sum().item()
    
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return accuracy, precision, recall, F1

#===============
#test loop
#===============
test_loss = 0
test_acc = 0
y_list = []
pred_labels = []
model.eval()
with torch.no_grad():
    for global_x, local_x, y in test_loader:
        global_x, local_x, y = global_x.to(device), local_x.to(device), y.to(device)
        
        pred = model(global_x, local_x)
        prob = torch.sigmoid(pred).view(-1)
        pred_label = (prob > 0.5).float()   
         
        loss = criterion(pred.view(-1), y)

        pred_labels.append(pred_label.detach().cpu())
        y_list.append(y.detach().cpu())
        
        test_loss += loss.item() * y.size(0)
        test_acc += (pred_label == y).float().sum().item()
        
    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
#evaluate the model
accuracy, precision, recall, F1 = evaluate(pred_labels, y_list)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {F1:.4f}")
    