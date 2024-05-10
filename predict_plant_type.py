 
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import cv2
from PIL import Image
import subprocess as sub
import secrets
import string
import os

num_classes = 39

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(secrets.choice(characters) for i in range(length))
    return random_string

    
def predict_type():
    model = CNNModel(num_classes)  

    model.load_state_dict(torch.load('plant/plant.pth'))

    model.eval() 
   
    image_path = 'uploads/image.png' 
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image_pil = Image.fromarray(image)


    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = transform(image_pil)
    input_image = input_image.unsqueeze(0)  

    data_dir = 'plant/data/'
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    with torch.no_grad():
        output = model(input_image)

    _, predicted = torch.max(output, 1)
    print(output)

    class_index = predicted.item()
    class_label = dataset.classes[class_index] 

    print(f'Tahmin edilen sınıf: {class_label}')
    filename = generate_random_string(20)
    folder_name = class_label
    folder_name = folder_name.replace(" ","\ ")
    class_label = class_label.replace("__"," ")
    class_label = class_label.replace("_"," ")
    print(folder_name)
    try:
        if not os.path.exists(f"plant/uploads/{folder_name}"):
            os.makedirs(f"plant/uploads/{folder_name}")
            print(f"{folder_name} has been created.")
        move_command = f"mv uploads/image.png plant/uploads/{folder_name}/{filename}.png"
        sub.run(move_command, shell=True)
    except:
        print("opps :(")
    return class_label