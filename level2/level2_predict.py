import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, 'cnn_model.pth')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'test_images')

#描述模型->导入参数
class cnn(nn.Module):
    def __init__(self):
        super(cnn,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128),nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x

def get_mnist_test_loader(batch_size=64, num_samples=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=True)

    if num_samples is not None:
        num_samples = min(num_samples, len(test_dataset))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(num_samples)))

    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def evaluate_accuracy(num_samples=1000, batch_size=64):
    model=cnn()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
    model.eval()

    loader=get_mnist_test_loader(batch_size=batch_size, num_samples=num_samples)

    correct=0
    total=0
    with torch.no_grad():
        for images,labels in loader:
            outputs=model(images)
            preds=torch.argmax(outputs,dim=1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)

    accuracy=correct/total if total>0 else 0.0
    print(f"Accuracy on {total} test samples: {accuracy:.4f} ({correct}/{total})")
    return accuracy

def show_predictions():
    model=cnn()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
    model.eval()

    loader=get_mnist_test_loader(batch_size=1)

    for i,(image,label) in enumerate(loader):
        with torch.no_grad():
            output=model(image)
            pred=torch.argmax(output,dim=1).item()
            prob=torch.softmax(output,dim=1)[0][pred].item()
        print(f"Sample {i}: True Label: {label.item()}, Predicted Label: {pred},confidence{prob:.4f}")

        if i>=9:
            break

if __name__=="__main__":
    evaluate_accuracy(num_samples=1000, batch_size=64)
    show_predictions()