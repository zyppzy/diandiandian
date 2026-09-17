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
MODEL_PATH = os.path.join(BASE_DIR, 'cnn_model.pth')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))]
)

train_data=datasets.MNIST(root=DATA_DIR,train=True,transform=transform,download=True)
train_loader=DataLoader(dataset=train_data,batch_size=64,shuffle=True)

images,labels=next(iter(train_loader))
print(f"shape of images: {images.shape}")
print(f"shape of labels: {labels.shape}")
print(f"labels: {labels[:5]}")


class cnn(nn.Module):
    def __init__(self):
        super(cnn,self).__init__()

        #定义结构，提取特征
        self.features=nn.Sequential(
            #第一个二维卷积核
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),#图像像素尺寸减半

            #第二个二维卷积核
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)#继续减半，变成7*7
        )

        #分类马，最终输出0-9中的数字
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128),nn.ReLU(),
            #随机关闭一半的神经元，防止过拟合
            nn.Dropout(p=0.3),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x

model=cnn()
print(model)
total_params=sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

#损失函数，优化器
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0002)

num_epochs=15
loss_history=[]
for epoch in range(num_epochs):
    model.train()
    total_loss=0.0
    for images,labels in train_loader:
        #前向传播
        outputs=model(images)
        loss=criterion(outputs,labels)
        #反向传播,更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    avg_loss=total_loss/len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

#绘制loss曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), loss_history, marker='o', label='Training Loss')
plt.title('Training Loss Curve CNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
OUTPUT_DIR = os.path.join(BASE_DIR, 'cnn_loss_curve.png')
plt.savefig(OUTPUT_DIR)
print(f"Loss curve saved to {OUTPUT_DIR}") 

#save the model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")