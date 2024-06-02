import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import RegressionModel
from data import dataset, split_data
from train import train_and_eval

# Define the dataset
X, y = dataset()
X_train, X_test, y_train, y_test = split_data(X, y)

# Define model parameters
input_size = X.shape[1]
output_size = 1
lr = 0.01

# Demonstrate double descent phenomenon with varying model complexities
model_complexities = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
all_train_losses = []
all_test_losses = []

for complexity in model_complexities:
    model = RegressionModel(input_size, output_size, complexity)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    train_losses, test_losses = train_and_eval(model, criterion, optimizer, X_train, y_train, X_test, y_test)
    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)

# Aggregate losses to observe double descent more clearly
avg_train_losses = [sum(train_loss)/len(train_loss) for train_loss in all_train_losses]
avg_test_losses = [sum(test_loss)/len(test_loss) for test_loss in all_test_losses]

# Plot average training and test losses vs model complexity with reduced scale
plt.figure(figsize=(8, 6))
plt.plot(model_complexities, avg_train_losses, marker='o', label='Avg Training Loss')
plt.plot(model_complexities, avg_test_losses, marker='o', label='Avg Test Loss')
plt.title('Average Training and Test Loss vs Model complexity')
plt.xlabel('Model complexity')
plt.ylabel('Avg Loss')
plt.ylim(0, max(avg_test_losses) * 1.2) 
plt.legend()
plt.show()
