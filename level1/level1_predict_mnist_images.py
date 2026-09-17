import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, 'mlp_model.pth')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'mnist_prediction_samples')

class mlp(nn.Module):
    def __init__(self):
        super(mlp,self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(28*28,256),nn.ReLU(),
            nn.Linear(256,128),nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        if x.dim()>2:
            x=x.view(x.size(0),-1)
        return self.layers(x)


def get_mnist_test_loader(batch_size=64, num_samples=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=True)

    if num_samples is not None:
        num_samples = min(num_samples, len(test_dataset))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(num_samples)))

    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def evaluate_accuracy(num_samples=1000, batch_size=64):
    model = mlp()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
    model.eval()

    loader = get_mnist_test_loader(batch_size=batch_size, num_samples=num_samples)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy on {total} test samples: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def show_predictions():
    model = mlp()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
    model.eval()

    loader = get_mnist_test_loader(batch_size=1)

    for idx, (images, labels) in enumerate(loader):
        if idx >= 10:
            break

        with torch.no_grad():
            output = model(images)

        pred = torch.argmax(output, dim=1).item()
        true_label = labels.item()
        prob = torch.softmax(output, dim=1)[0][pred].item()
        print(f"sample {idx}: true={true_label}, pred={pred}, confidence={prob:.4f}")


if __name__ == "__main__":
    evaluate_accuracy(num_samples=1000, batch_size=64)
    show_predictions()  

