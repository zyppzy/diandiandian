import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#定义对图片的加工流程
#ToTenser()将图片转化为张量，将0-255的像素值转化为0-1之间的浮点数
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: torch.flatten(x))]#此处的x用于承接已经转换为张量的数据
    )

#下载训练集，使用定义过的transform对图片进行加工
train_data=datasets.MNIST(root='./data',train=True,transform=transform,download=True)   

#打包数据，同一批次输入64张图片的数据，并打乱
train_loader=DataLoader(dataset=train_data,batch_size=64,shuffle=True)

#检验
"""
images,labels=next(iter(train_loader))
print(f"shape of images: {images.shape}")
print(f"shape of labels: {labels.shape}")
print(f"labels: {labels[:5]}")
"""

class mlp(nn.Module):
    def __init__(self):
        super(mlp,self).__init__()
        #输入层->隐藏层->隐藏层->输出层
        self.layers=nn.Sequential(
            nn.Linear(28*28,256),nn.ReLU(),
            nn.Linear(256,128),nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        return self.layers(x)

#实例化
model=mlp()
print(model)
total_params=sum(p.numel() for p in model.parameters())
#model.parameters()表示张量 p.numel()表示其中的参数个数
print(f"Total parameters: {total_params}")

#损失函数
criterion=nn.CrossEntropyLoss()
#优化器调参参数
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

#训练循环
num_epochs=10
loss_history=[]
for epoch in range(num_epochs):
    model.train()
    total_loss=0.0
    for images,labels in train_loader:
        #前向传播
        outputs=model(images)
        loss=criterion(outputs,labels)
        #反向传播,更新参数
        optimizer.zero_grad()#上一轮的梯度清零
        loss.backward()
        optimizer.step()#更新参数
        total_loss+=loss.item()

    avg_loss=total_loss/len(train_loader)
    loss_history.append(avg_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

#绘制loss曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
plt.title('Model Training Loss Curve')
plt.xlabel('Epoch')                    
plt.ylabel('Loss')                    
plt.grid(True)                     
plt.savefig('loss_curve.png')          

#保存模型
torch.save(model.state_dict(), 'mlp_model.pth')
print("Model saved as 'mlp_model.pth'") 