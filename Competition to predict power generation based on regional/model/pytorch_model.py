# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 02:40:33 2024

@author: User
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()

        # Define each layer of the neural network
        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, 1)

        # Initialize weights
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        
        self.batch_size = 32
        self.epochs = 1
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        
    # Define forward propagation
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer for regression
        return x
    
    def fit(self, X_train, Y_train, verbose=True):
        self.to(self.device)
        
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            # Set model to training mode
            self.train()
        
            running_loss = 0.0
            total_samples = 0

            # Iterate over batches
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                self.optimizer.zero_grad()
                Y_pred = self(X_batch)
                loss = self.criterion(Y_pred, Y_batch)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * X_batch.size(0)
                total_samples += X_batch.size(0)


            epoch_loss = running_loss / total_samples
            
            if verbose:
                print(f'Epoch {epoch+1}/{self.epochs},MSE Loss: {epoch_loss:.4f}')    

    def predict(self, X_test):
        self.to(self.device)  
        self.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
            Y_pred = self(X_test_tensor)
        
        return Y_pred.cpu().numpy()
    
# In[]


class RegressionModel2(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel2, self).__init__()

        # Define network layers with Batch Normalization
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 2)

        # Initialize weights and biases
        init.xavier_normal_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        init.xavier_normal_(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)
        init.xavier_normal_(self.fc3.weight)
        init.constant_(self.fc3.bias, 0)
        
        # Define Dropout layer
        self.dropout = nn.Dropout(0.3)

        # Training parameters
        self.batch_size = 32
        self.epochs = 10
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)  # Reduced learning rate
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def fit(self, X_train, Y_train, X_val=None, Y_val=None, verbose=True, patience=10):
        self.to(self.device)

        # Convert training data to tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).reshape(-1, Y_train.shape[1]).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        

        for epoch in range(self.epochs):
            self.train()
            running_loss = 0.0
            total_samples = 0

            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                self.optimizer.zero_grad()
                Y_pred = self(X_batch)

                # Check for NaNs in predictions
                if torch.isnan(Y_pred).any():
                    raise ValueError("NaNs detected in model predictions.")

                loss = self.criterion(Y_pred, Y_batch)

                # Check for NaNs in loss
                if torch.isnan(loss):
                    raise ValueError("NaN loss encountered.")

                loss.backward()

                # Check for NaNs in gradients
                for name, param in self.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        raise ValueError(f"NaNs detected in gradients of {name}.")

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                self.optimizer.step()

                running_loss += loss.item() * X_batch.size(0)
                total_samples += X_batch.size(0)

            epoch_loss = running_loss / total_samples
            self.scheduler.step()

            if verbose:
                print(f'Epoch {epoch+1}/{self.epochs}, Training MSE Loss: {epoch_loss:.4f}')


    def predict(self, X_test):
        self.to(self.device)
        self.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
            Y_pred = self(X_test_tensor)
        
        return Y_pred.cpu().numpy()
    
# In[]
class GRURegressionModel(nn.Module):
    def __init__(self, input_size):
        super(GRURegressionModel, self).__init__()
        self.hidden_size = 100
        self.num_layers = 2

        self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, 1)

        self.batch_size = 32
        self.epochs = 10
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

    def fit(self, X_train, Y_train, verbose=True):
        self.to(self.device)

        # 确保输入数据的形状为 (batch_size, seq_len, input_size)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.train()
            running_loss = 0.0
            total_samples = 0

            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                self.optimizer.zero_grad()
                Y_pred = self(X_batch)
                loss = self.criterion(Y_pred, Y_batch)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * X_batch.size(0)
                total_samples += X_batch.size(0)

            epoch_loss = running_loss / total_samples
            if verbose:
                print(f'Epoch {epoch+1}/{self.epochs}, MSE Loss: {epoch_loss:.4f}')

    def predict(self, X_test):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            Y_pred = self(X_test_tensor)
        return Y_pred.cpu().numpy()
