import os
import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, 'alexnet_model.pth')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


#输入图像尺寸为 224x224 的3通道RGB彩色图像（原始训练图像为256x256，随机裁剪为224x224
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))]
)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # 适配 28x28 MNIST 图像的 AlexNet 轻量版
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(96, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        dummy_input = torch.randn(1, 1, 28, 28)
        flat_size = self.features(dummy_input).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data=datasets.MNIST(root=DATA_DIR,train=True,transform=transform,download=True)
    train_loader=DataLoader(dataset=train_data,batch_size=64,shuffle=True)

    images,labels=next(iter(train_loader))
    print(f"shape of images: {images.shape}")
    print(f"shape of labels: {labels.shape}")
    print(f"labels: {labels[:5]}")

    model=AlexNet().to(device)
    print(model)
    total_params=sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0002)

    num_epochs=10
    loss_history=[]
    for epoch in range(num_epochs):
        model.train()
        total_loss=0.0
        for images,labels in train_loader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            loss=criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

        avg_loss=total_loss/len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), loss_history, marker='o', label='Training Loss')
    plt.title('Training Loss Curve AlexNet')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    OUTPUT_DIR = os.path.join(BASE_DIR, 'alexnet_loss_curve.png')
    plt.savefig(OUTPUT_DIR)
    print(f"Loss curve saved to {OUTPUT_DIR}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")