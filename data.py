import torch
from sklearn.model_selection import train_test_split


def dataset(samples=500, features=1000, noise_level=0.1):
    X = torch.randn(samples, features)
    y = 3 * X[:, 0] + 0.1 * torch.randn(samples)
    return X, y.unsqueeze(1)

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)