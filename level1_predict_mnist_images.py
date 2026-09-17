import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mlp_model.pth')
DATA_DIR = os.path.join(BASE_DIR, 'data')
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


def get_mnist_test_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=True)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def show_predictions():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = mlp()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
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

        image = images[0].clone().reshape(28, 28)
        img = image.numpy()

        print(f"sample {idx}: true={true_label}, pred={pred}, confidence={prob:.4f}")

        plt.figure(figsize=(2, 2))
        plt.imshow(img, cmap='gray')
        plt.title(f'true={true_label}, pred={pred}', fontsize=8)
        plt.axis('off')

        save_path = os.path.join(OUTPUT_DIR, f'mnist_sample_{idx}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"saved image: {save_path}")

    print(f"All sample images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    show_predictions()  

