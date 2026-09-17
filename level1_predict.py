import os
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mlp_model.pth')
TEST_IMAGES_DIR = os.path.join(BASE_DIR, 'test_images')

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
            #x.view()将输入张量x的形状调整为(batch_size, 28*28)
        return self.layers(x)   

def prepare_mnist_image(image_path):
    image = Image.open(image_path).convert('L')
    image = ImageOps.autocontrast(image)
    image = ImageOps.pad(image, (max(image.size), max(image.size)), color=255)
    image = image.resize((28, 28), Image.BILINEAR)

    image_array = np.array(image, dtype=np.float32)
    if image_array.mean() > 127:
        image_array = 255.0 - image_array

    image_tensor = torch.from_numpy(image_array / 255.0)
    image_tensor = image_tensor.flatten()
    image_tensor = (image_tensor - 0.1307) / 0.3081
    return image_tensor.unsqueeze(0)


def predict_image(image_path, model_path=MODEL_PATH):
    model=mlp()

    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded model from: {model_path}")
    except FileNotFoundError:
        print(f"Model file '{model_path}' not found. ")

    model.eval()  #设置模型为评估模式

    input_image = prepare_mnist_image(image_path)

    with torch.no_grad():
        output=model(input_image)

    _,predicted=torch.max(output,1)  #获取最大值的索引
    predicted_label=predicted.item()  #将张量转换为Python的标量

    predicted_probabilities=torch.softmax(output,dim=1)[0]*100

    print(f"Predicted label: {predicted_label}")

    for i in range(10):
        print(f"Probability of {i}: {predicted_probabilities[i].item():.2f}%")

if __name__=="__main__":
    image_path = os.path.join(TEST_IMAGES_DIR, 'image2.png')
    print(f"Testing image: {image_path}")
    predict_image(image_path)  

