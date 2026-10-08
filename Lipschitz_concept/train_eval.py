import torch
import torch.nn as nn
import numpy as np

##### UTILITY FUNCTIONS #####
# The following functions are just utility functions to avoid code repetition. Their format is based on personal preference

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train() 
    train_loss = 0.0
    total_samples = 0

    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad() # Must be set to zero each iteration to avoid exploding gradients
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward() # Backpropagation to compute gradients
        optimizer.step() # Update the model parameters

    return

def evaluate(model, dataloader, loss_fn, device):
    model.eval() # Set the model to evaluation mode, this makes the model behave faster as no dropout, batchnorm, or gradient calculations are needed
    val_loss = 0.0
    total_samples = 0

    with torch.no_grad(): # No gradient computation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            val_loss += loss_fn(y_pred, y).item()*X.size(0) # Multiply by batch size to get the total loss for this batch
            total_samples += X.size(0)

    return val_loss / total_samples

def train(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, max_iter,  device):
    train_losses = []
    val_losses = []

    early_stopping_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_loss = evaluate(model, train_loader, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= max_iter:
            print("Early stopping triggered.")
            break

    return train_losses, val_losses

def predict(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            y_pred = model(X)
            predictions.append(y_pred.cpu().numpy())
    return np.concatenate(predictions, axis=0)