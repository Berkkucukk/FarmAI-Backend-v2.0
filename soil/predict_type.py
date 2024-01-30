import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import cv2
from PIL import Image
import subprocess as sub
import secrets
import string

num_classes = 5

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Giriş boyutunu uygun şekilde ayarlayın
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
    # Model yapısını oluştur
    model = CNNModel(num_classes)  # Model yapısını aynı şekilde oluşturmalısınız

    # Kaydedilmiş parametreleri yükle
    model.load_state_dict(torch.load('soil.pth'))

    # Modeli tahminlerde kullanabilirsiniz
    model.eval()  # Modeli tahmin modunda etkinleştirin
    # Burada tahminlerinizi yapabilirsiniz


    image_path = 'uploads/image.png'  # Test görüntüsünün dosya yolu
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Görüntüyü BGR'den RGB'ye dönüştür
    image_pil = Image.fromarray(image)


    # Giriş görüntüsünü modele uygun hale getir
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = transform(image_pil)
    input_image = input_image.unsqueeze(0)  # Batch boyutunu ekleyin (1 görüntü için)

    # Veri yolu - Klasördeki verilerin kategori alt klasörlerde bulunduğunu varsayalım.
    data_dir = 'Soil types/'
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    with torch.no_grad():
        output = model(input_image)

    # Sınıflandırma sonuçlarını al
    _, predicted = torch.max(output, 1)
    print(output)
    # Sınıf tahminini bul
    class_index = predicted.item()
    class_label = dataset.classes[class_index]  # 'dataset' veri kümesinin sınıflarını içerdiğini varsayalım

    print(f'Tahmin edilen sınıf: {class_label}')
    filename = generate_random_string(20)
    folder_name = class_label
    folder_name = folder_name.replace(" ","\ ")
    print(folder_name)
    try:
        move_command = f"cp uploads/image.png uploads/{folder_name}/{filename}.png"
        sub.run(move_command, shell=True)
    except:
        print("opps :(")
    return class_label