import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cv2
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

source_transform = transforms.Compose([
    # 轉灰階: Canny 不吃 RGB。
    transforms.Grayscale(),
    # cv2 不吃 skimage.Image，因此轉成np.array後再做cv2.Canny
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # 重新將np.array 轉回 skimage.Image
    transforms.ToPILImage(),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15, fill=(0,)),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((32, 32)),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15, fill=(0,)),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])




class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # input [1, 32, 32]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # [512, ]
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y


source_dataset = ImageFolder('../real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('../real_or_drawing/test_data', transform=target_transform)


source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)


feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()


feature_extractor.load_state_dict(torch.load('extractor_model.bin'))
label_predictor.load_state_dict(torch.load('predictor_model.bin'))


class_criterion = nn.CrossEntropyLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())



def train_epoch(source_dataloader, test_dataloader):
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    for i, (source_data, source_label) in enumerate(source_dataloader):

        source_data = source_data.cuda()
        source_label = source_label.cuda()

        feature = feature_extractor(source_data)

        class_logits = label_predictor(feature)

        loss = class_criterion(class_logits, source_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        print(i, end='\r')


for epoch in range(1000):
    label_predictor.train()
    feature_extractor.train()
    train_epoch(source_dataloader, test_dataloader)

    torch.save(feature_extractor.state_dict(), f'./models/extractor_model_{epoch}.bin')
    torch.save(label_predictor.state_dict(), f'./models/predictor_model_{epoch}.bin')

    result = np.zeros(100000)
    label_predictor.eval()
    feature_extractor.eval()
    num = np.zeros(10)
    cnt = 0
    for i, (test_data, _) in enumerate(test_dataloader):
        test_data = test_data.cuda()

        class_logits = label_predictor(feature_extractor(test_data))

        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        for q in x:
            result[cnt] = int(q)
            cnt += 1
            num[int(q)] +=  1
        print(i, end='\r')
    print(np.sum( abs(num - 10000) ))
    print(num)
    print(f'epoch : {epoch}')
    
    with open(f'./csvs/DaNN_submission_{epoch}.csv', 'w') as f:
        f.write('id,label\n')
        for i, y in  enumerate(result):
            f.write('{},{}\n'.format(i, int(y)))

