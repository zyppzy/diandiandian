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
MODEL_PATH = os.path.join(BASE_DIR, 'resnet_model.pth')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

class BasicBlock(nn.Module):
    expansion = 1  # 通道数不变

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample  # 尺寸/通道不匹配时的捷径

    def forward(self, x):
        identity=x
        out=self.features(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity #这一步是核心，加入了原始输入和映射
        out = nn.ReLU(inplace=True)(out)#给残差相加的结果引入非线性，激活，块的表达能力更强
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        #通道数逐渐翻倍，stride=2特征图尺寸减半
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    #建造残杀块，
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        #判断条件，缩小空间尺寸&输入输出通道不一致
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        #构建空链表layers来存放以及完成计算的残差块，
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    images, labels = next(iter(train_loader))
    print(f"shape of images: {images.shape}")
    print(f"shape of labels: {labels.shape}")
    print(f"labels: {labels[:5]}")

    model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
    print(model)

    total_params=sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0002)

    num_epochs=15
    loss_history=[]
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        total_loss=0.0
        print(f"\n===== Epoch [{epoch+1}/{num_epochs}] starts =====")
        for step, (images, labels) in enumerate(train_loader, start=1):
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            loss=criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

            # 每 100 个 batch 输出一次进度，最后一个 batch 也输出
            if step % 100 == 0 or step == total_steps:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}/{total_steps}] - Loss: {loss.item():.4f}")

        avg_loss=total_loss/len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Finished - Average Loss: {avg_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), loss_history, marker='o', label='Training Loss')
    plt.title('Training Loss Curve ResNet')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    OUTPUT_DIR = os.path.join(BASE_DIR, 'resnet_loss_curve.png')
    plt.savefig(OUTPUT_DIR)
    print(f"Loss curve saved to {OUTPUT_DIR}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    